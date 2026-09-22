"""CheXpert-style label cells -> (target, loss-mask) arrays.

A MIMIC-CXR-JPG label cell is one of:
    1.0  finding present
    0.0  finding explicitly negated
   -1.0  finding mentioned but uncertain
    NaN  finding never mentioned in the report

Every published number on this dataset depends on how the last two are handled,
so the policy is explicit, configurable, and written into the manifest.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..constants import LABELS

UNCERTAIN_POLICIES = ("ignore", "ones", "zeros", "per_class")
BLANK_POLICIES = ("zeros", "ignore")


def apply_policy(
    frame: pd.DataFrame,
    uncertain_policy: str = "ignore",
    per_class_policy: Dict[str, str] | None = None,
    blank_policy: str = "zeros",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y, mask), both float32 [N, 4].

    y[i, c]     target in {0.0, 1.0}
    mask[i, c]  1.0 when the cell contributes to the loss, 0.0 when ignored
    """
    if uncertain_policy not in UNCERTAIN_POLICIES:
        raise ValueError(f"uncertain_policy must be one of {UNCERTAIN_POLICIES}")
    if blank_policy not in BLANK_POLICIES:
        raise ValueError(f"blank_policy must be one of {BLANK_POLICIES}")

    missing = [name for name in LABELS if name not in frame.columns]
    if missing:
        raise KeyError(f"label columns missing from frame: {missing}")

    raw = frame[LABELS].to_numpy(dtype=np.float64, copy=True)
    y = np.zeros_like(raw, dtype=np.float32)
    mask = np.ones_like(raw, dtype=np.float32)

    is_nan = np.isnan(raw)
    is_unc = ~is_nan & (raw == -1.0)
    is_pos = ~is_nan & (raw == 1.0)

    y[is_pos] = 1.0

    if blank_policy == "ignore":
        mask[is_nan] = 0.0
    # blank_policy == "zeros": y already 0, mask already 1.

    for c, name in enumerate(LABELS):
        policy = uncertain_policy
        if uncertain_policy == "per_class":
            policy = (per_class_policy or {}).get(name, "ignore")
        col = is_unc[:, c]
        if not col.any():
            continue
        if policy == "ones":
            y[col, c] = 1.0
        elif policy == "zeros":
            y[col, c] = 0.0
        else:  # ignore
            mask[col, c] = 0.0

    return y, mask


def attach_targets(
    frame: pd.DataFrame,
    uncertain_policy: str = "ignore",
    per_class_policy: Dict[str, str] | None = None,
    blank_policy: str = "zeros",
) -> pd.DataFrame:
    """Add y_<label> and m_<label> columns to a manifest frame."""
    y, mask = apply_policy(frame, uncertain_policy, per_class_policy, blank_policy)
    out = frame.copy()
    for c, name in enumerate(LABELS):
        out[f"y_{name}"] = y[:, c]
        out[f"m_{name}"] = mask[:, c]
    return out


def targets_from_manifest(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Read back the y_/m_ columns written by `attach_targets`."""
    y = frame[[f"y_{name}" for name in LABELS]].to_numpy(dtype=np.float32)
    mask = frame[[f"m_{name}" for name in LABELS]].to_numpy(dtype=np.float32)
    return y, mask


def label_stats(y: np.ndarray, mask: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Per-class prevalence over the cells that actually count."""
    stats: Dict[str, Dict[str, float]] = {}
    for c, name in enumerate(LABELS):
        valid = mask[:, c] > 0
        n_valid = int(valid.sum())
        n_pos = int(y[valid, c].sum()) if n_valid else 0
        stats[name] = {
            "n_labelled": n_valid,
            "n_positive": n_pos,
            "prevalence": (n_pos / n_valid) if n_valid else 0.0,
            "n_ignored": int((~valid).sum()),
        }
    return stats


def pos_weight_from(y: np.ndarray, mask: np.ndarray, cap: float = 10.0) -> np.ndarray:
    """BCE pos_weight = negatives / positives per class, clamped for stability."""
    weights = np.ones(y.shape[1], dtype=np.float32)
    for c in range(y.shape[1]):
        valid = mask[:, c] > 0
        pos = float(y[valid, c].sum())
        neg = float(valid.sum() - pos)
        if pos > 0:
            weights[c] = float(np.clip(neg / pos, 1.0 / cap, cap))
    return weights


def combo_key(y_row: np.ndarray) -> str:
    """'1010' style key for a label combination - used for stratified sampling."""
    return "".join("1" if v > 0.5 else "0" for v in y_row)
