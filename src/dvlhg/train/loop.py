"""Stage 2: train the vision-language backbone (encoders + fusion + head).

Everything here is built for Colab's failure mode: the session dies at 90
minutes. Each epoch writes a full checkpoint (model, optimiser, scheduler,
scaler, EMA, history), and `train_backbone(..., resume=True)` picks up exactly
where it stopped. Put `paths.root` on Drive and a disconnect costs one epoch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import DotDict, save_config
from ..constants import LABELS
from ..data.dataset import make_loaders, ordered_loader
from ..data.labels import pos_weight_from, targets_from_manifest
from ..eval.metrics import compute_metrics, tune_thresholds
from ..models.dvlhgn import Backbone
from ..utils import (
    EMA,
    AverageMeter,
    amp_enabled,
    cosine_warmup_lambda,
    count_params,
    get_logger,
    human,
    load_checkpoint,
    pick_device,
    save_checkpoint,
    save_json,
    set_seed,
)
from ..vlm.encoders import build_vlm
from .losses import build_loss

LOG = get_logger()

CKPT_NAME = "backbone.pt"
BEST_NAME = "backbone_best.pt"


def build_backbone(cfg: DotDict, device: torch.device) -> Tuple[Backbone, object]:
    vlm, tokenize = build_vlm(cfg)
    model = Backbone(vlm, cfg).to(device)
    model.vlm.set_trainable(
        vision_blocks=int(cfg.vlm.unfreeze_vision_blocks),
        text_blocks=int(cfg.vlm.unfreeze_text_blocks),
        train_norms=bool(cfg.vlm.train_vision_norm),
    )
    # Fusion, heads and the hypergraph read-out are new modules: always trained.
    for module in (model.fusion, model.head, model.image_head, model.text_head):
        for param in module.parameters():
            param.requires_grad = True
    total, trainable = count_params(model)
    LOG.info("backbone: %s params, %s trainable", human(total), human(trainable))
    return model, tokenize


@torch.no_grad()
def predict(
    model: Backbone,
    loader,
    device: torch.device,
    use_amp: bool = False,
    collect_nodes: bool = False,
) -> Dict[str, np.ndarray]:
    model.eval()
    logits, image_logits, text_logits = [], [], []
    targets, masks, nodes, gates = [], [], [], []
    img_embeds, txt_embeds = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=use_amp):
            out = model(images, input_ids, attention_mask)
        logits.append(out["logits"].float().cpu().numpy())
        image_logits.append(out["image_logits"].float().cpu().numpy())
        text_logits.append(out["text_logits"].float().cpu().numpy())
        gates.append(out["gate"].float().cpu().numpy())
        targets.append(batch["y"].numpy())
        masks.append(batch["mask"].numpy())
        if collect_nodes:
            nodes.append(out["node"].float().cpu().numpy())
            img_embeds.append(out["img_pooled"].float().cpu().numpy())
            txt_embeds.append(out["txt_pooled"].float().cpu().numpy())

    result = {
        "logits": np.concatenate(logits),
        "image_logits": np.concatenate(image_logits),
        "text_logits": np.concatenate(text_logits),
        "gate": np.concatenate(gates),
        "y": np.concatenate(targets),
        "mask": np.concatenate(masks),
    }
    if collect_nodes:
        result["node"] = np.concatenate(nodes)
        result["img_pooled"] = np.concatenate(img_embeds)
        result["txt_pooled"] = np.concatenate(txt_embeds)
    return result


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def evaluate(
    model: Backbone, loader, device: torch.device, use_amp: bool, thresholds: Optional[np.ndarray] = None
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    out = predict(model, loader, device, use_amp)
    probabilities = _sigmoid(out["logits"])
    if thresholds is None:
        thresholds = tune_thresholds(out["y"], probabilities, out["mask"], "f1")
    metrics = compute_metrics(out["y"], probabilities, out["mask"], thresholds)
    return metrics, out


def train_backbone(
    cfg: DotDict,
    manifest: pd.DataFrame,
    images: Dict[str, np.ndarray],
    resume: bool = True,
    diffusion_refiner=None,
) -> Path:
    set_seed(int(cfg.seed))
    device = pick_device(cfg.get("device", "auto"))
    use_amp = amp_enabled(bool(cfg.train.amp), device)
    ckpt_dir = Path(cfg.paths.ckpt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / CKPT_NAME
    best_path = ckpt_dir / BEST_NAME

    model, tokenize = build_backbone(cfg, device)
    loaders = make_loaders(manifest, images, tokenize, cfg)
    if "train" not in loaders:
        raise RuntimeError("no training data")

    y_train, mask_train = targets_from_manifest(manifest[manifest["split"] == "train"])
    pos_weight = None
    if str(cfg.train.pos_weight) == "auto":
        pos_weight = torch.from_numpy(pos_weight_from(y_train, mask_train))
        LOG.info("pos_weight: %s", dict(zip(LABELS, np.round(pos_weight.numpy(), 2).tolist())))
    elif isinstance(cfg.train.pos_weight, (list, tuple)):
        pos_weight = torch.tensor([float(v) for v in cfg.train.pos_weight])
    criterion = build_loss(cfg, pos_weight).to(device)

    groups = model.trainable_parameter_groups(
        lr=float(cfg.train.lr),
        encoder_lr_scale=float(cfg.train.encoder_lr_scale),
        weight_decay=float(cfg.train.weight_decay),
    )
    optimizer = torch.optim.AdamW(groups)
    steps_per_epoch = max(len(loaders["train"]) // max(int(cfg.train.accum_steps), 1), 1)
    total_steps = steps_per_epoch * int(cfg.train.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_warmup_lambda(total_steps, int(total_steps * float(cfg.train.warmup_ratio)))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, decay=float(cfg.train.ema_decay)) if float(cfg.train.ema_decay) > 0 else None

    start_epoch, best_score, bad_epochs = 0, -np.inf, 0
    history: List[dict] = []
    state: Optional[dict] = None
    if resume and ckpt_path.exists():
        state = load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        if ema and state.get("ema"):
            ema.load_state_dict(state["ema"])
        start_epoch = int(state.get("epoch", 0))
        best_score = float(state.get("best_score", -np.inf))
        bad_epochs = int(state.get("bad_epochs", 0))
        history = list(state.get("history", []))
        LOG.info("resumed backbone training at epoch %d (best %.4f)", start_epoch, best_score)

    epochs = int(cfg.train.epochs)
    accum = max(int(cfg.train.accum_steps), 1)
    refine_prob = float(cfg.diffusion.refine_prob) if diffusion_refiner is not None else 0.0

    for epoch in range(start_epoch, epochs):
        model.train()
        loss_meter, main_meter = AverageMeter(), AverageMeter()
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(loaders["train"], desc=f"backbone {epoch + 1}/{epochs}", leave=False)
        except ImportError:
            iterator = loaders["train"]

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(iterator):
            batch_images = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            weight = batch["weight"].to(device, non_blocking=True)

            if refine_prob > 0 and np.random.rand() < refine_prob:
                batch_images = diffusion_refiner(batch_images, targets)

            with torch.autocast("cuda", enabled=use_amp):
                out = model(batch_images, input_ids, attention_mask)
                main = criterion(out["logits"], targets, mask, weight)
                aux_image = criterion(out["image_logits"], targets, mask, weight)
                aux_text = criterion(out["text_logits"], targets, mask, weight)
                loss = main + 0.3 * aux_image + 0.2 * aux_text

            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                if float(cfg.train.grad_clip) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], float(cfg.train.grad_clip)
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if ema:
                    ema.update(model)

            loss_meter.update(float(loss.detach()), batch_images.shape[0])
            main_meter.update(float(main.detach()), batch_images.shape[0])
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{loss_meter.avg:.4f}")

        if ema:
            ema.apply_to(model)
        val_metrics, _ = evaluate(model, loaders.get("val", loaders["train"]), device, use_amp)
        if ema:
            ema.restore(model)

        score = float(val_metrics.get(str(cfg.train.monitor).split("/")[-1], val_metrics["auroc_macro"]))
        record = {
            "epoch": epoch + 1,
            "train_loss": loss_meter.avg,
            "train_main_loss": main_meter.avg,
            "val_auroc_macro": val_metrics["auroc_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_auprc_macro": val_metrics["auprc_macro"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)
        LOG.info(
            "epoch %d/%d - loss %.4f | val AUROC %.4f | val F1 %.4f",
            epoch + 1, epochs, loss_meter.avg, val_metrics["auroc_macro"], val_metrics["f1_macro"],
        )

        improved = score > best_score
        if improved:
            best_score, bad_epochs = score, 0
        else:
            bad_epochs += 1

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.state_dict() if ema else None,
            "epoch": epoch + 1,
            "best_score": best_score,
            "bad_epochs": bad_epochs,
            "history": history,
            "config": cfg.to_dict(),
            "val_metrics": val_metrics,
            "vlm_backend": model.vlm.backend,
        }
        if bool(cfg.train.save_every_epoch):
            save_checkpoint(state, ckpt_path)
        if improved:
            save_checkpoint(state, best_path)
            LOG.info("new best (%s = %.4f) -> %s", cfg.train.monitor, best_score, best_path.name)

        if bad_epochs >= int(cfg.train.early_stop_patience):
            LOG.info("early stop: %d epochs without improvement", bad_epochs)
            break

    save_json(history, Path(cfg.paths.logs) / "backbone_history.json")
    save_config(cfg, ckpt_dir / "backbone_config.yaml")
    if not best_path.exists() and state is not None:
        save_checkpoint(state, best_path)
    if not best_path.exists():
        raise RuntimeError(
            "no checkpoint was written - train.epochs may be <= the resumed epoch count"
        )
    LOG.info("best backbone checkpoint: %s", best_path)
    return best_path


def load_backbone(
    cfg: DotDict, device: Optional[torch.device] = None, which: str = "best"
) -> Tuple[Backbone, object, dict]:
    device = device or pick_device(cfg.get("device", "auto"))
    path = Path(cfg.paths.ckpt) / (BEST_NAME if which == "best" else CKPT_NAME)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `dvlhg train` first")
    state = load_checkpoint(path, map_location=device)
    saved_cfg = DotDict(state.get("config", cfg.to_dict()))
    # Keep runtime paths/devices from the live config, architecture from the checkpoint.
    saved_cfg.paths = cfg.paths
    model, tokenize = build_backbone(saved_cfg, device)
    model.load_state_dict(state["model"])
    model.eval()
    LOG.info("loaded backbone from %s (epoch %s)", path.name, state.get("epoch"))
    return model, tokenize, state


@torch.no_grad()
def extract_features(
    cfg: DotDict,
    model: Backbone,
    manifest: pd.DataFrame,
    images: Dict[str, np.ndarray],
    tokenize,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """One deterministic pass over every row -> the node features the
    hypergraph is built on. Row order matches the manifest exactly."""
    device = device or pick_device(cfg.get("device", "auto"))
    use_amp = amp_enabled(bool(cfg.train.amp), device)
    loader = ordered_loader(manifest, images, tokenize, cfg)
    out = predict(model, loader, device, use_amp, collect_nodes=True)
    LOG.info("extracted node features: %s", out["node"].shape)
    return out
