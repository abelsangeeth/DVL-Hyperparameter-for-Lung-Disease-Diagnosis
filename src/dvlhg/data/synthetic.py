"""A procedurally generated stand-in for MIMIC-CXR.

This is not a substitute for real data and produces no publishable numbers. It
exists so that the entire pipeline — preprocessing, diffusion, the VLM, fusion,
the hypergraph, evaluation, export and the API — can be run and tested on a
laptop in a couple of minutes, with no credentials and no downloads.

The films carry genuine, label-correlated structure (an enlarged cardiac
silhouette, a blunted costophrenic angle, a basal band, perihilar haze), so a
working model reaches ~0.85-0.95 AUC here and a broken one does not. That makes
it a real regression test rather than a shape check.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from ..constants import LABELS
from ..utils import get_logger
from .labels import attach_targets

LOG = get_logger()

# Weak, realistic priors: the reason-for-exam nudges the label distribution but
# never determines it, so the text channel is informative without being a giveaway.
INDICATIONS: Dict[str, List[str]] = {
    "Atelectasis": ["status post abdominal surgery", "decreased breath sounds at the left base", "post-operative day 2, hypoxia"],
    "Cardiomegaly": ["history of congestive heart failure", "hypertension, evaluate cardiac size", "known cardiomyopathy"],
    "Edema": ["acute shortness of breath", "chf exacerbation", "volume overload on dialysis"],
    "Pleural Effusion": ["dyspnea, evaluate for fluid", "decreased breath sounds on the right", "cirrhosis with ascites"],
    "_none": ["routine pre-operative chest radiograph", "line placement, confirm position", "screening examination", "cough and low grade fever"],
}
TECHNIQUES = ["chest pa and lateral", "portable ap chest", "chest ap upright", "chest pa"]


def _grid(size: int) -> Tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    return np.meshgrid(axis, axis)  # xx, yy


def _ellipse(xx, yy, cx, cy, rx, ry, rot=0.0):
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    x = (xx - cx) * cos_r + (yy - cy) * sin_r
    y = -(xx - cx) * sin_r + (yy - cy) * cos_r
    return (x / rx) ** 2 + (y / ry) ** 2


def _soft(mask: np.ndarray, hardness: float = 12.0) -> np.ndarray:
    # Clipped: the ellipse field is unbounded far from its centre and would
    # otherwise overflow exp() on every call.
    return 1.0 / (1.0 + np.exp(np.clip((mask - 1.0) * hardness, -60.0, 60.0)))


def render_cxr(labels: np.ndarray, size: int = 256, rng: np.random.Generator | None = None) -> np.ndarray:
    """Render one synthetic frontal film for a 4-vector of labels."""
    rng = rng or np.random.default_rng()
    xx, yy = _grid(size)
    atel, cmeg, edem, peff = (float(v) for v in labels)

    img = np.full((size, size), 0.04, dtype=np.float32)

    # Thorax soft tissue.
    body = _soft(_ellipse(xx, yy, 0.0, 0.05, 0.82, 0.95), 8.0)
    img += 0.34 * body

    # Lung fields (radiolucent -> dark). Atelectasis costs a little volume.
    lung_h = 0.60 - 0.05 * atel
    jitter = rng.normal(0, 0.015, 4)
    left = _soft(_ellipse(xx, yy, -0.36 + jitter[0], -0.02 + jitter[1], 0.28, lung_h), 10.0)
    right = _soft(_ellipse(xx, yy, 0.36 + jitter[2], -0.02 + jitter[3], 0.28, lung_h), 10.0)
    lungs = np.clip(left + right, 0, 1)
    img -= 0.30 * lungs

    # Mediastinum + cardiac silhouette. Cardiomegaly widens it (higher CTR).
    heart_rx = 0.17 + 0.13 * cmeg + rng.normal(0, 0.012)
    heart = _soft(_ellipse(xx, yy, -0.05, 0.28, heart_rx, 0.26), 11.0)
    spine = _soft(_ellipse(xx, yy, 0.0, 0.0, 0.055, 0.92), 14.0)
    img += 0.26 * heart + 0.18 * spine

    # Ribs.
    for i in range(9):
        y0 = -0.62 + i * 0.15
        arc = np.exp(-((yy - y0 - 0.16 * xx**2) ** 2) / 0.00055)
        img += 0.055 * arc * body

    # Diaphragm.
    diaphragm = _soft(_ellipse(xx, yy, 0.0, 1.05, 1.05, 0.55), 9.0)
    img += 0.22 * diaphragm

    # --- findings -----------------------------------------------------------
    if peff > 0.5:
        side = rng.choice([-1.0, 1.0])
        height = 0.18 + 0.12 * rng.random()
        wedge = _soft(_ellipse(xx, yy, side * 0.42, 0.72, 0.30, height), 10.0)
        meniscus = np.clip(1.0 - np.exp(-((yy - (0.72 - height)) ** 2) / 0.0012), 0, 1)
        img += 0.30 * wedge * (0.65 + 0.35 * meniscus)

    if atel > 0.5:
        side = rng.choice([-1.0, 1.0])
        band = _soft(_ellipse(xx, yy, side * 0.34, 0.42, 0.22, 0.10, rot=side * 0.35), 12.0)
        img += 0.26 * band * lungs

    if edem > 0.5:
        haze = np.exp(-(_ellipse(xx, yy, -0.18, 0.05, 0.45, 0.42)) * 1.6)
        haze += np.exp(-(_ellipse(xx, yy, 0.18, 0.05, 0.45, 0.42)) * 1.6)
        texture = rng.normal(0, 1, (size, size)).astype(np.float32)
        texture = _blur(texture, 2) * 0.10
        img += 0.16 * haze * lungs + texture * haze * lungs

    # --- acquisition variation ---------------------------------------------
    img += rng.normal(0, 0.018, (size, size)).astype(np.float32)
    img = np.clip(img, 0, 1) ** float(rng.uniform(0.85, 1.20))          # gamma
    img = np.clip((img - 0.5) * float(rng.uniform(0.9, 1.15)) + 0.5, 0, 1)  # contrast
    return img.astype(np.float32)


def _blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Separable box-blur approximation of a Gaussian (no scipy dependency)."""
    radius = max(int(sigma), 1)
    kernel = np.ones(2 * radius + 1, dtype=np.float32)
    kernel /= kernel.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, arr)
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, out)
    return out


def make_report(labels: np.ndarray, rng: np.random.Generator) -> str:
    """A MIMIC-shaped report: honest pre-read context plus a findings body."""
    positives = [name for name, value in zip(LABELS, labels) if value > 0.5]
    pool: List[str] = []
    for name in positives:
        if rng.random() < 0.7:  # the referrer does not always guess right
            pool.append(rng.choice(INDICATIONS[name]))
    if not pool or rng.random() < 0.25:
        pool.append(rng.choice(INDICATIONS["_none"]))
    indication = "; ".join(dict.fromkeys(pool))

    findings = []
    findings.append("the cardiac silhouette is enlarged." if labels[1] > 0.5 else "the cardiomediastinal silhouette is normal in size.")
    if labels[3] > 0.5:
        findings.append("there is a small to moderate pleural effusion with blunting of the costophrenic angle.")
    else:
        findings.append("no pleural effusion or pneumothorax.")
    if labels[0] > 0.5:
        findings.append("there is basilar atelectasis.")
    if labels[2] > 0.5:
        findings.append("there is diffuse interstitial pulmonary edema with vascular congestion.")
    if not positives:
        findings.append("the lungs are clear.")

    impression = ", ".join(positives).lower() if positives else "no acute cardiopulmonary process"
    return (
        "                                 FINAL REPORT\n"
        f" EXAMINATION:  CHEST RADIOGRAPH\n\n"
        f" INDICATION:  ___ year old patient with {indication}\n\n"
        f" TECHNIQUE:  {rng.choice(TECHNIQUES)}\n\n"
        f" COMPARISON:  {'none' if rng.random() < 0.5 else 'prior chest radiograph'}\n\n"
        f" FINDINGS: \n {' '.join(findings)}\n\n"
        f" IMPRESSION: \n {impression}.\n"
    )


def build_dataset(
    raw_dir: str | os.PathLike,
    n: int = 600,
    size: int = 256,
    seed: int = 1337,
    val_fraction: float = 0.15,
    test_fraction: float = 0.20,
) -> pd.DataFrame:
    """Generate images + reports on disk and return the manifest."""
    rng = np.random.default_rng(seed)
    raw = Path(raw_dir)
    (raw / "images").mkdir(parents=True, exist_ok=True)

    # Prevalences and a positive correlation between edema and effusion, which
    # is what makes the hypergraph's co-occurrence structure meaningful.
    base_rates = np.array([0.30, 0.25, 0.20, 0.28], dtype=np.float32)
    rows: List[dict] = []
    for i in range(n):
        labels = (rng.random(4) < base_rates).astype(np.float32)
        if labels[2] > 0.5 and rng.random() < 0.55:
            labels[3] = 1.0
        if labels[3] > 0.5 and rng.random() < 0.35:
            labels[0] = 1.0

        image = render_cxr(labels, size=size, rng=rng)
        name = f"synth_{i:06d}.png"
        Image.fromarray((image * 255).astype(np.uint8), mode="L").save(raw / "images" / name)

        record = {
            "dicom_id": f"synth_{i:06d}",
            "subject_id": f"sp{i // 3:05d}",   # ~3 studies per patient
            "study_id": f"ss{i:06d}",
            "rel_path": f"images/{name}",
            "ViewPosition": "PA" if rng.random() < 0.7 else "AP",
            "sex": "M" if rng.random() < 0.5 else "F",
            "age": int(rng.integers(25, 90)),
            "report_text": make_report(labels, rng),
            "report_rel_path": "",
            "source": "synthetic",
        }
        for name_, value in zip(LABELS, labels):
            record[name_] = float(value)
        rows.append(record)

    frame = pd.DataFrame(rows)
    frame = attach_targets(frame, "ignore", None, "zeros")

    # Split by patient so the same synthetic subject never spans folds.
    patients = frame["subject_id"].unique()
    order = rng.permutation(len(patients))
    n_test = int(round(len(patients) * test_fraction))
    n_val = int(round(len(patients) * val_fraction))
    assignment = {}
    for rank, idx in enumerate(order):
        if rank < n_test:
            assignment[patients[idx]] = "test"
        elif rank < n_test + n_val:
            assignment[patients[idx]] = "val"
        else:
            assignment[patients[idx]] = "train"
    frame["split"] = frame["subject_id"].map(assignment)

    LOG.info(
        "synthetic dataset: %d films (%s)",
        len(frame),
        ", ".join(f"{k}={v}" for k, v in frame["split"].value_counts().items()),
    )
    return frame
