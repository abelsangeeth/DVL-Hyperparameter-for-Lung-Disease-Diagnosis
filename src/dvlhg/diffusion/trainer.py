"""Stage 1: train the label-conditional diffusion model.

Trained on the TRAINING SPLIT ONLY. A generator that has seen validation or
test films would launder those images back into training through the synthetic
set, and every number after that would be meaningless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import DotDict, save_config
from ..constants import LABELS
from ..data.labels import targets_from_manifest
from ..utils import (
    EMA,
    AverageMeter,
    amp_enabled,
    count_params,
    get_logger,
    human,
    load_checkpoint,
    pick_device,
    save_checkpoint,
    save_json,
    set_seed,
)
from .ddpm import GaussianDiffusion, to_image_space, to_model_space
from .unet import ConditionalUNet

LOG = get_logger()


class DiffusionImages(Dataset):
    """Films at the diffusion resolution, in [-1, 1], with their label vector."""

    def __init__(self, images: np.ndarray, indices: np.ndarray, labels: np.ndarray, size: int):
        self.images = images
        self.indices = indices
        self.labels = labels.astype(np.float32)
        self.size = int(size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        row = int(self.indices[i])
        array = np.ascontiguousarray(self.images[row])
        tensor = torch.from_numpy(array).float().div_(255.0).unsqueeze(0)
        if tensor.shape[-1] != self.size:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(self.size, self.size), mode="bilinear", align_corners=False
            ).squeeze(0)
        if np.random.rand() < 0.5:
            tensor = _jitter(tensor)
        return to_model_space(tensor), torch.from_numpy(self.labels[i])


def _jitter(image: torch.Tensor) -> torch.Tensor:
    contrast = float(np.random.uniform(0.92, 1.08))
    return ((image - 0.5) * contrast + 0.5).clamp(0, 1)


def build_diffusion(cfg: DotDict, device: torch.device) -> GaussianDiffusion:
    unet = ConditionalUNet(
        image_size=int(cfg.diffusion.image_size),
        in_channels=int(cfg.diffusion.channels),
        base_channels=int(cfg.diffusion.base_channels),
        channel_mult=tuple(int(m) for m in cfg.diffusion.channel_mult),
        num_res_blocks=int(cfg.diffusion.num_res_blocks),
        attn_resolutions=tuple(int(r) for r in cfg.diffusion.attn_resolutions),
        num_labels=len(LABELS),
    )
    diffusion = GaussianDiffusion(
        unet,
        timesteps=int(cfg.diffusion.timesteps),
        schedule=str(cfg.diffusion.schedule),
        objective=str(cfg.diffusion.objective),
    ).to(device)
    total, trainable = count_params(unet)
    LOG.info("diffusion UNet: %s params (%s trainable)", human(total), human(trainable))
    return diffusion


def train_diffusion(
    cfg: DotDict,
    manifest: pd.DataFrame,
    images: np.ndarray,
    resume: bool = True,
) -> Path:
    set_seed(int(cfg.seed))
    device = pick_device(cfg.get("device", "auto"))
    ckpt_dir = Path(cfg.paths.ckpt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "diffusion.pt"

    train_rows = manifest["split"] == "train"
    if not train_rows.any():
        raise RuntimeError("no training rows - cannot train the diffusion model")
    indices = manifest.loc[train_rows, "cache_index"].to_numpy(dtype=np.int64)
    y, _ = targets_from_manifest(manifest.loc[train_rows])
    LOG.info("training diffusion on %d films from the TRAIN split only", len(indices))

    dataset = DiffusionImages(images, indices, y, int(cfg.diffusion.image_size))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.diffusion.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=int(cfg.data.num_workers) > 0,
    )

    diffusion = build_diffusion(cfg, device)
    optimizer = torch.optim.AdamW(
        diffusion.model.parameters(), lr=float(cfg.diffusion.train.lr), weight_decay=0.0
    )
    ema = EMA(diffusion.model, decay=float(cfg.diffusion.train.ema_decay))
    use_amp = amp_enabled(bool(cfg.diffusion.train.amp), device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    if resume and ckpt_path.exists():
        state = load_checkpoint(ckpt_path, map_location=device)
        diffusion.model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        start_epoch = int(state.get("epoch", 0))
        LOG.info("resumed diffusion training from epoch %d", start_epoch)

    epochs = int(cfg.diffusion.train.epochs)
    history = []
    for epoch in range(start_epoch, epochs):
        diffusion.train()
        meter = AverageMeter()
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(loader, desc=f"diffusion {epoch + 1}/{epochs}", leave=False)
        except ImportError:
            iterator = loader

        for batch_images, batch_labels in iterator:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss = diffusion.loss(
                    batch_images, batch_labels, cond_dropout=float(cfg.diffusion.cond_dropout)
                )
            scaler.scale(loss).backward()
            if float(cfg.diffusion.train.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    diffusion.model.parameters(), float(cfg.diffusion.train.grad_clip)
                )
            scaler.step(optimizer)
            scaler.update()
            ema.update(diffusion.model)
            meter.update(float(loss.detach()), batch_images.shape[0])
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{meter.avg:.4f}")

        LOG.info("diffusion epoch %d/%d - loss %.4f", epoch + 1, epochs, meter.avg)
        history.append({"epoch": epoch + 1, "loss": meter.avg})
        save_checkpoint(
            {
                "model": diffusion.model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "config": cfg.to_dict(),
                "history": history,
            },
            ckpt_path,
        )

    save_json(history, Path(cfg.paths.logs) / "diffusion_history.json")
    save_config(cfg, Path(cfg.paths.ckpt) / "diffusion_config.yaml")
    LOG.info("diffusion checkpoint at %s", ckpt_path)
    return ckpt_path


def load_diffusion(
    cfg: DotDict, device: Optional[torch.device] = None, use_ema: bool = True
) -> GaussianDiffusion:
    device = device or pick_device(cfg.get("device", "auto"))
    ckpt_path = Path(cfg.paths.ckpt) / "diffusion.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found - run `dvlhg diffusion` first")
    state = load_checkpoint(ckpt_path, map_location=device)
    saved = DotDict(state.get("config", cfg.to_dict()))
    diffusion = build_diffusion(saved, device)
    diffusion.model.load_state_dict(state["model"])
    if use_ema and "ema" in state:
        shadow = state["ema"]["shadow"]
        with torch.no_grad():
            for name, param in diffusion.model.named_parameters():
                if name in shadow:
                    param.copy_(shadow[name].to(param.dtype).to(param.device))
        LOG.info("loaded EMA weights for sampling")
    diffusion.eval()
    return diffusion


@torch.no_grad()
def sample_grid(
    diffusion: GaussianDiffusion, cfg: DotDict, out_path: Path, n_per_class: int = 4
) -> Path:
    """One row per finding - a quick visual check that samples look like films."""
    device = next(diffusion.model.parameters()).device
    rows = []
    label_sets = [np.zeros(len(LABELS), np.float32)] + [
        np.eye(len(LABELS), dtype=np.float32)[i] for i in range(len(LABELS))
    ]
    for vector in label_sets:
        labels = torch.from_numpy(np.tile(vector, (n_per_class, 1))).to(device)
        samples = diffusion.ddim_sample(
            shape=(n_per_class, 1, int(cfg.diffusion.image_size), int(cfg.diffusion.image_size)),
            labels=labels,
            steps=int(cfg.diffusion.sample_steps),
            guidance_scale=float(cfg.diffusion.guidance_scale),
            device=device,
        )
        rows.append(to_image_space(samples).cpu().numpy()[:, 0])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = ["no finding"] + LABELS
        fig, axes = plt.subplots(len(rows), n_per_class, figsize=(n_per_class * 2.0, len(rows) * 2.1))
        axes = np.atleast_2d(axes)
        for r, (row, name) in enumerate(zip(rows, names)):
            for c in range(n_per_class):
                axes[r, c].imshow(row[c], cmap="gray", vmin=0, vmax=1)
                axes[r, c].axis("off")
            axes[r, 0].set_title(name, loc="left", fontsize=9)
        fig.suptitle("Diffusion samples — SYNTHETIC, not real patient images", fontsize=10)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        LOG.info("sample grid written to %s", out_path)
    except ImportError:
        LOG.warning("matplotlib unavailable - skipping the sample grid")
    return out_path
