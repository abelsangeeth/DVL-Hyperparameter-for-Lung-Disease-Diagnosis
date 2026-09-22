"""Explanations for a single prediction.

Three complementary views, all cheap enough to run inside a web request:

  saliency   Grad-CAM over the fused image tokens: where on the film the
             evidence for one finding sits.
  text       cross-attention mass per report token: which words the image
             attended to.
  neighbours the hyperedge members nearest the query — case-based evidence,
             which is the one explanation a hypergraph model gets for free.

A caveat worth repeating in any write-up: saliency maps show where a model
looked, not whether it was right to. They are a debugging aid, not evidence of
clinical validity.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..constants import LABELS


def grad_cam(
    model,
    image: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    class_index: int,
) -> Tuple[Optional[np.ndarray], Dict[str, torch.Tensor]]:
    """Return ([H, W] map in [0, 1] or None, the fusion extras from the same pass).

    Returning the extras matters: the caller needs the cross-attention weights
    too, and running the backbone a second time to get them both doubles the
    latency and leaves a second graph alive.
    """
    was_training = model.training
    model.eval()

    # Make the input require grad rather than touching parameter flags: a frozen
    # backbone would otherwise build no graph at all, and flipping requires_grad
    # on the model would silently break any later training.
    with torch.enable_grad():
        image = image.detach().clone().requires_grad_(True)
        out = model(image, input_ids, attention_mask, return_extras=True)
        extras = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in out.get("extras", {}).items()
        }
        tokens = out.get("extras", {}).get("image_tokens")
        if tokens is None:
            if was_training:
                model.train()
            return None, extras
        logit = out["logits"][0, class_index]
        grads = torch.autograd.grad(logit, tokens, retain_graph=False, allow_unused=True)[0]
        heat = None if grads is None else _cam_from(tokens, grads, image.shape[-1])

    if was_training:
        model.train()
    return heat, extras


def grad_cam_multi(
    model,
    image: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    class_indices: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], Dict[str, torch.Tensor]]:
    """Grad-CAM for several findings from a single forward pass.

    One forward, then one backward per finding with `retain_graph=True`. Running
    `grad_cam` four times instead would repeat the encoder four times, which is
    the expensive half -- this costs roughly one forward plus three cheap
    backwards.
    """
    was_training = model.training
    model.eval()
    maps: Dict[int, np.ndarray] = {}

    with torch.enable_grad():
        image = image.detach().clone().requires_grad_(True)
        out = model(image, input_ids, attention_mask, return_extras=True)
        extras = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in out.get("extras", {}).items()
        }
        tokens = out.get("extras", {}).get("image_tokens")
        if tokens is None:
            if was_training:
                model.train()
            return maps, extras

        indices = list(class_indices)
        for position, class_index in enumerate(indices):
            grads = torch.autograd.grad(
                out["logits"][0, class_index],
                tokens,
                retain_graph=position < len(indices) - 1,
                allow_unused=True,
            )[0]
            if grads is None:
                continue
            heat = _cam_from(tokens, grads, image.shape[-1])
            if heat is not None:
                maps[int(class_index)] = heat

    if was_training:
        model.train()
    return maps, extras


def _cam_from(tokens: torch.Tensor, grads: torch.Tensor, size: int) -> Optional[np.ndarray]:
    """Weight patch activations by their mean gradient, reshape to the grid."""
    with torch.no_grad():
        weights = grads.mean(dim=1, keepdim=True)                 # [1, 1, d]
        cam = torch.relu((tokens * weights).sum(dim=-1))[0]        # [Ni]
        patches = cam[1:] if cam.numel() > 1 else cam              # drop CLS
        side = int(math.isqrt(patches.numel()))
        if side * side != patches.numel():
            return None
        heat = torch.nn.functional.interpolate(
            patches.view(1, 1, side, side), size=(size, size),
            mode="bilinear", align_corners=False,
        )[0, 0]
        heat = heat - heat.min()
        heat = heat / heat.max().clamp_min(1e-8)
    return heat.detach().float().cpu().numpy()


def text_importance(
    extras: Dict[str, torch.Tensor], tokens: Sequence[str], top_k: int = 12
) -> List[Dict[str, float]]:
    """Average cross-attention mass the image placed on each report token."""
    attention = extras.get("text_attention")
    if attention is None:
        return []
    weights = attention[0].mean(dim=0).detach().float().cpu().numpy()  # [Nt]
    limit = min(len(tokens), len(weights))
    scored = [
        {"token": tokens[i], "weight": float(weights[i])}
        for i in range(limit)
        if tokens[i] not in ("[PAD]", "[CLS]", "[SEP]", "")
    ]
    scored.sort(key=lambda item: -item["weight"])
    top = scored[:top_k]
    if top:
        peak = max(item["weight"] for item in top) or 1.0
        for item in top:
            item["weight"] = round(item["weight"] / peak, 4)
    return top


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _colormap(values: np.ndarray) -> np.ndarray:
    """Black-red-yellow-white ramp. Self-contained: no matplotlib at serve time."""
    v = np.clip(values, 0.0, 1.0)
    red = np.clip(v * 3.0, 0, 1)
    green = np.clip(v * 3.0 - 1.0, 0, 1)
    blue = np.clip(v * 3.0 - 2.0, 0, 1)
    return np.stack([red, green, blue], axis=-1)


def overlay_heatmap(
    image: np.ndarray, heat: np.ndarray, alpha: float = 0.45, threshold: float = 0.35
) -> np.ndarray:
    """Blend a heatmap over a grayscale film -> uint8 RGB."""
    base = np.clip(image, 0, 1)
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    colour = _colormap(heat)
    strength = np.clip((heat - threshold) / max(1.0 - threshold, 1e-6), 0, 1)[..., None]
    blended = base * (1 - alpha * strength) + colour * (alpha * strength)
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def to_png_base64(array: np.ndarray) -> str:
    """uint8 HxW or HxWx3 -> data URI, ready for an <img src>."""
    from PIL import Image

    if array.dtype != np.uint8:
        array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    mode = "L" if array.ndim == 2 else "RGB"
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def neighbour_records(
    neighbour_indices: Sequence[int],
    similarities: Sequence[float],
    manifest_rows: Sequence[dict],
    labels: Sequence[str] = LABELS,
    limit: int = 6,
) -> List[Dict[str, object]]:
    """Package the nearest hyperedge members for the UI."""
    records: List[Dict[str, object]] = []
    for position, (index, similarity) in enumerate(zip(neighbour_indices, similarities)):
        if position >= limit:
            break
        row = manifest_rows[int(index)] if int(index) < len(manifest_rows) else {}
        findings = [name for name in labels if float(row.get(f"y_{name}", 0.0)) > 0.5]
        records.append(
            {
                "index": int(index),
                "similarity": round(float(similarity), 4),
                "split": str(row.get("split", "")),
                "view": str(row.get("ViewPosition", "")),
                "findings": findings or ["no target finding"],
                "synthetic": str(row.get("cache_source", "real")) == "synth",
                "study": str(row.get("study_id", "")),
            }
        )
    return records
