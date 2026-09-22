"""Turn the trained diffusion model into extra training rows.

Label vectors are drawn from the *training* label distribution conditioned on
each finding in turn, which keeps co-occurrence realistic (edema rarely appears
without something else) while giving each of the four findings an equal budget.
That is the point of the exercise: the rare combinations a 12k subset barely
covers get the most synthetic support.

Every synthetic row is tagged `cache_source == "synth"`, lands in the training
split only, and carries `train.synthetic_weight` in the loss. Nothing synthetic
ever reaches validation or test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import DotDict
from ..constants import LABELS
from ..data.labels import combo_key, targets_from_manifest
from ..utils import get_logger, pick_device, save_json
from .ddpm import GaussianDiffusion, to_image_space
from .trainer import load_diffusion

LOG = get_logger()

SYNTH_CACHE = "synth_u8.npy"
SYNTH_MANIFEST = "synth_manifest.csv"


def _sample_label_vectors(
    train_labels: np.ndarray, per_class: int, rng: np.random.Generator
) -> np.ndarray:
    """For each finding, resample real training label vectors that contain it."""
    chosen: List[np.ndarray] = []
    for c in range(train_labels.shape[1]):
        pool = train_labels[train_labels[:, c] > 0.5]
        if len(pool) == 0:
            LOG.warning("no training positives for %s - skipping its synthetic budget", LABELS[c])
            continue
        picks = rng.integers(0, len(pool), size=per_class)
        chosen.append(pool[picks])
    if not chosen:
        raise RuntimeError("no label vectors to synthesise from")
    return np.concatenate(chosen, axis=0).astype(np.float32)


def _text_pool(manifest: pd.DataFrame) -> Tuple[Dict[str, List[str]], List[str]]:
    """Training reports indexed by label combination, for pairing with samples."""
    train = manifest[manifest["split"] == "train"]
    y, _ = targets_from_manifest(train)
    pool: Dict[str, List[str]] = {}
    for text, row in zip(train["text"].astype(str), y):
        pool.setdefault(combo_key(row), []).append(text)
    return pool, list(train["text"].astype(str))


def generate_synthetic(
    cfg: DotDict,
    manifest: pd.DataFrame,
    diffusion: Optional[GaussianDiffusion] = None,
    batch_size: int = 32,
) -> Tuple[pd.DataFrame, Path]:
    """Sample films, write the synthetic cache, return the synthetic manifest."""
    device = pick_device(cfg.get("device", "auto"))
    diffusion = diffusion or load_diffusion(cfg, device)
    diffusion.eval()
    rng = np.random.default_rng(int(cfg.seed) + 7)

    train = manifest[manifest["split"] == "train"]
    train_y, _ = targets_from_manifest(train)
    per_class = int(cfg.diffusion.synth_per_class)
    if per_class <= 0:
        raise ValueError("diffusion.synth_per_class must be > 0")

    vectors = _sample_label_vectors(train_y, per_class, rng)
    total = len(vectors)
    cache_size = int(cfg.data.cache_size)
    LOG.info("generating %d synthetic films at %dpx (stored at %dpx)",
             total, int(cfg.diffusion.image_size), cache_size)

    out_dir = Path(cfg.paths.synth)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / SYNTH_CACHE
    memmap = np.lib.format.open_memmap(
        cache_file, mode="w+", dtype=np.uint8, shape=(total, cache_size, cache_size)
    )

    try:
        from tqdm.auto import tqdm

        batches = tqdm(range(0, total, batch_size), desc="synthesis", unit="batch")
    except ImportError:
        batches = range(0, total, batch_size)

    for start in batches:
        stop = min(start + batch_size, total)
        labels = torch.from_numpy(vectors[start:stop]).to(device)
        samples = diffusion.ddim_sample(
            shape=(stop - start, 1, int(cfg.diffusion.image_size), int(cfg.diffusion.image_size)),
            labels=labels,
            steps=int(cfg.diffusion.sample_steps),
            guidance_scale=float(cfg.diffusion.guidance_scale),
            device=device,
        )
        images = to_image_space(samples)
        if images.shape[-1] != cache_size:
            images = torch.nn.functional.interpolate(
                images, size=(cache_size, cache_size), mode="bilinear", align_corners=False
            )
        memmap[start:stop] = (images[:, 0].clamp(0, 1) * 255).round().byte().cpu().numpy()
    memmap.flush()
    del memmap

    # ---- manifest ----------------------------------------------------------
    pool, fallback = _text_pool(manifest)
    views = train["ViewPosition"].astype(str).to_numpy() if "ViewPosition" in train.columns else np.array(["PA"])
    mode = str(cfg.diffusion.get("synth_text", "sample_train"))

    rows: List[dict] = []
    for i, vector in enumerate(vectors):
        key = combo_key(vector)
        if mode == "placeholder" or (key not in pool and not fallback):
            text = str(cfg.text.empty_placeholder)
        else:
            candidates = pool.get(key) or fallback
            text = str(candidates[int(rng.integers(0, len(candidates)))])
        record = {
            "dicom_id": f"synth_{i:06d}",
            "subject_id": "SYNTHETIC",
            "study_id": f"synth_{i:06d}",
            "rel_path": "",
            "report_rel_path": "",
            "ViewPosition": str(views[int(rng.integers(0, len(views)))]),
            "split": "train",
            "text": text,
            "cache_index": i,
            "cache_source": "synth",
            "source": "diffusion",
        }
        for c, name in enumerate(LABELS):
            record[f"y_{name}"] = float(vector[c])
            record[f"m_{name}"] = 1.0
            record[name] = float(vector[c])
        rows.append(record)

    synth_manifest = pd.DataFrame(rows)
    manifest_file = out_dir / SYNTH_MANIFEST
    synth_manifest.to_csv(manifest_file, index=False)

    counts = {name: int(synth_manifest[f"y_{name}"].sum()) for name in LABELS}
    save_json(
        {
            "total": total,
            "per_class_requested": per_class,
            "positives_generated": counts,
            "image_size": int(cfg.diffusion.image_size),
            "cache_size": cache_size,
            "guidance_scale": float(cfg.diffusion.guidance_scale),
            "sample_steps": int(cfg.diffusion.sample_steps),
            "text_mode": mode,
        },
        out_dir / "synth_stats.json",
    )
    LOG.info("wrote %d synthetic rows to %s (positives: %s)", total, manifest_file, counts)
    return synth_manifest, cache_file


def load_synthetic(cfg: DotDict) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """Return (manifest, image memmap) if a synthetic set exists, else (None, None)."""
    out_dir = Path(cfg.paths.synth)
    manifest_file = out_dir / SYNTH_MANIFEST
    cache_file = out_dir / SYNTH_CACHE
    if not manifest_file.exists() or not cache_file.exists():
        return None, None
    frame = pd.read_csv(manifest_file)
    frame["text"] = frame["text"].fillna("").astype(str)
    return frame, np.load(cache_file, mmap_mode="r")


def merge_with_synthetic(
    manifest: pd.DataFrame, cfg: DotDict
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], bool]:
    """Attach the synthetic training rows, if any, to the real manifest."""
    from ..data.prepare import load_image_cache

    real_images = load_image_cache(cfg)
    manifest = manifest.copy()
    if "cache_source" not in manifest.columns:
        manifest["cache_source"] = "real"
    images: Dict[str, np.ndarray] = {"real": real_images}

    if not bool(cfg.train.use_synthetic):
        return manifest, images, False

    synth_manifest, synth_images = load_synthetic(cfg)
    if synth_manifest is None or synth_images is None:
        LOG.info("no synthetic set found - training on real films only")
        return manifest, images, False

    columns = [c for c in manifest.columns if c in synth_manifest.columns]
    missing = [c for c in manifest.columns if c not in synth_manifest.columns]
    for column in missing:
        synth_manifest[column] = "" if manifest[column].dtype == object else 0
    merged = pd.concat([manifest, synth_manifest[manifest.columns]], axis=0, ignore_index=True)
    images["synth"] = synth_images
    LOG.info(
        "training set: %d real + %d synthetic rows (synthetic loss weight %.2f)",
        len(manifest), len(synth_manifest), float(cfg.train.synthetic_weight),
    )
    return merged, images, True
