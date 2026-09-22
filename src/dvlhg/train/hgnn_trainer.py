"""Stage 3: build the hypergraph over frozen node features and train the HGNN.

Full-batch, so an epoch is two sparse mat-muls — a few milliseconds. Three
hundred epochs take less time than one backbone epoch, which is what makes it
practical to actually ablate the hypergraph design rather than assert that it
helps.

Only training-node labels enter the loss. Validation and test nodes take part in
message passing (the standard transductive HGNN setting) but their labels are
never read; `hypergraph.transductive: false` drops them from the graph entirely
if you need a strictly inductive comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import DotDict, save_config
from ..data.labels import pos_weight_from, targets_from_manifest
from ..eval.metrics import compute_metrics, tune_thresholds
from ..hypergraph.construct import Hypergraph, build_hypergraph
from ..models.dvlhgn import HypergraphClassifier
from ..utils import get_logger, load_checkpoint, pick_device, save_checkpoint, save_json, set_seed
from .losses import build_loss

LOG = get_logger()

CKPT_NAME = "hypergraph.pt"


def _meta_from_manifest(manifest: pd.DataFrame) -> Dict[str, np.ndarray]:
    meta: Dict[str, np.ndarray] = {}
    if "ViewPosition" in manifest.columns:
        meta["view"] = manifest["ViewPosition"].fillna("UNKNOWN").astype(str).to_numpy()
    if "sex" in manifest.columns and "age" in manifest.columns:
        meta["sex"] = manifest["sex"].fillna("U").astype(str).to_numpy()
        meta["age"] = pd.to_numeric(manifest["age"], errors="coerce").to_numpy(dtype=np.float64)
    return meta


def make_graph(
    cfg: DotDict, manifest: pd.DataFrame, features: Dict[str, np.ndarray], device: torch.device
) -> Tuple[Hypergraph, torch.Tensor, np.ndarray]:
    """Build the hypergraph and return (graph, node features, row selector)."""
    split = manifest["split"].astype(str).to_numpy()
    keep = np.ones(len(manifest), dtype=bool)
    if not bool(cfg.hypergraph.transductive):
        keep = split == "train"
        LOG.info("inductive mode: hypergraph built from %d training nodes only", int(keep.sum()))

    node = torch.from_numpy(features["node"][keep]).float().to(device)
    image = torch.from_numpy(features["img_pooled"][keep]).float().to(device)
    text = torch.from_numpy(features["txt_pooled"][keep]).float().to(device)
    y, mask = targets_from_manifest(manifest[keep])
    meta = {k: v[keep] for k, v in _meta_from_manifest(manifest).items()}

    graph = build_hypergraph(
        fused=node, image=image, text=text,
        labels=y, mask=mask, split=split[keep], meta=meta, cfg=cfg, seed=int(cfg.seed),
    )
    return graph, node, keep


def train_hypergraph(
    cfg: DotDict,
    manifest: pd.DataFrame,
    features: Dict[str, np.ndarray],
    resume: bool = False,
) -> Tuple[HypergraphClassifier, Hypergraph, Dict[str, object]]:
    set_seed(int(cfg.seed))
    device = pick_device(cfg.get("device", "auto"))
    graph, node_features, keep = make_graph(cfg, manifest, features, device)

    sub_manifest = manifest[keep].reset_index(drop=True)
    split = sub_manifest["split"].astype(str).to_numpy()
    y_all, mask_all = targets_from_manifest(sub_manifest)
    targets = torch.from_numpy(y_all).float().to(device)
    cell_mask = torch.from_numpy(mask_all).float().to(device)

    index = {name: np.where(split == name)[0] for name in ("train", "val", "test")}
    train_idx = torch.from_numpy(index["train"]).long().to(device)
    if len(index["val"]) == 0:
        raise RuntimeError("no validation nodes - cannot early stop the HGNN")
    val_idx = torch.from_numpy(index["val"]).long().to(device)

    # The backbone's own logits are the residual base the HGNN corrects.
    base_logits = torch.from_numpy(features["logits"][keep]).float().to(device)

    model = HypergraphClassifier(node_dim=node_features.shape[1], cfg=cfg).to(device)
    # bind_graph allocates the per-edge weight residual; it must exist before
    # the optimiser is constructed or those parameters never get updated.
    model.encoder.bind_graph(graph)
    model.to(device)

    pos_weight = torch.from_numpy(pos_weight_from(y_all[index["train"]], mask_all[index["train"]]))
    criterion = build_loss(cfg, pos_weight).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.hgnn_train.lr), weight_decay=float(cfg.hgnn_train.weight_decay)
    )

    ckpt_path = Path(cfg.paths.ckpt) / CKPT_NAME
    best_score, best_state, bad_epochs = -np.inf, None, 0
    history = []
    sample_weight = torch.ones(len(sub_manifest), device=device)
    if "cache_source" in sub_manifest.columns:
        synthetic = torch.from_numpy(
            (sub_manifest["cache_source"].astype(str).to_numpy() == "synth").astype(np.float32)
        ).to(device)
        sample_weight = torch.where(
            synthetic > 0,
            torch.full_like(synthetic, float(cfg.train.synthetic_weight)),
            torch.ones_like(synthetic),
        )

    epochs = int(cfg.hgnn_train.epochs)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(node_features, graph, base_logits)
        loss = criterion(
            logits[train_idx], targets[train_idx], cell_mask[train_idx], sample_weight[train_idx]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits = model(node_features, graph, base_logits)
        probabilities = torch.sigmoid(eval_logits).cpu().numpy()
        val_metrics = compute_metrics(
            y_all[index["val"]], probabilities[index["val"]], mask_all[index["val"]],
            tune_thresholds(y_all[index["val"]], probabilities[index["val"]], mask_all[index["val"]], "f1"),
        )
        score = float(val_metrics["auroc_macro"])
        history.append(
            {"epoch": epoch + 1, "loss": float(loss.detach()),
             "val_auroc_macro": score, "val_f1_macro": val_metrics["f1_macro"]}
        )

        if score > best_score:
            best_score, bad_epochs = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            LOG.info(
                "hgnn epoch %d/%d - loss %.4f | val AUROC %.4f (best %.4f)",
                epoch + 1, epochs, float(loss.detach()), score, best_score,
            )
        if bad_epochs >= int(cfg.hgnn_train.early_stop_patience):
            LOG.info("hgnn early stop at epoch %d (best val AUROC %.4f)", epoch + 1, best_score)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        final = model(node_features, graph, base_logits)
        delta = float((final - base_logits).abs().mean())
    LOG.info(
        "mean |logit correction| from the hypergraph: %.4f (delta_scale %.3f)",
        delta, float(model.delta_scale.detach()),
    )

    weights = model.encoder.group_weights().detach().cpu().numpy()
    from ..constants import EDGE_GROUPS

    LOG.info(
        "learned hyperedge group weights: %s",
        {name: round(float(w), 3) for name, w in zip(EDGE_GROUPS, weights)},
    )

    save_checkpoint(
        {
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "history": history,
            "best_val_auroc": best_score,
            "graph_summary": graph.summary(),
            "node_dim": int(node_features.shape[1]),
            "keep_mask": keep,
        },
        ckpt_path,
    )
    save_json(history, Path(cfg.paths.logs) / "hgnn_history.json")
    save_config(cfg, Path(cfg.paths.ckpt) / "hypergraph_config.yaml")

    info = {
        "best_val_auroc": best_score,
        "graph": graph.summary(),
        "group_weights": {name: float(w) for name, w in zip(EDGE_GROUPS, weights)},
        "checkpoint": str(ckpt_path),
        "mean_logit_correction": delta,
        "delta_scale": float(model.delta_scale.detach()),
        "keep": keep,
        "sub_manifest_len": int(len(sub_manifest)),
    }
    return model, graph, info


@torch.no_grad()
def hypergraph_predictions(
    model: HypergraphClassifier,
    node_features: torch.Tensor,
    graph: Hypergraph,
    base_logits: torch.Tensor,
) -> np.ndarray:
    model.eval()
    return model(node_features, graph, base_logits).float().cpu().numpy()


def load_hypergraph_model(
    cfg: DotDict, graph: Hypergraph, node_dim: int, device: Optional[torch.device] = None
) -> HypergraphClassifier:
    device = device or pick_device(cfg.get("device", "auto"))
    path = Path(cfg.paths.ckpt) / CKPT_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `dvlhg hypergraph` first")
    state = load_checkpoint(path, map_location=device)
    model = HypergraphClassifier(node_dim=node_dim, cfg=DotDict(state.get("config", cfg.to_dict()))).to(device)
    model.encoder.bind_graph(graph)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if missing:
        LOG.warning("hypergraph checkpoint is missing keys: %s", missing)
    if unexpected:
        LOG.warning("hypergraph checkpoint has unexpected keys: %s", unexpected)
    model.eval()
    return model
