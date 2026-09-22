"""Stage 0: build the manifest and the image cache that every later stage reads.

Output under paths.manifest / paths.cache:
    manifest.csv      one row per study: paths, split, y_*/m_* targets, text
    images_u8.npy     [N, S, S] uint8 memmap of preprocessed films
    stats.json        prevalences, split sizes, text mode, policies

The image cache is one big memmap rather than N small files on purpose: copying
12k JPEGs to Google Drive takes the better part of an hour, copying a single
800 MB file takes a couple of minutes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

from ..config import DotDict, ensure_dirs
from ..constants import LABELS
from ..utils import get_logger, save_json
from . import mimic, openi, synthetic
from .labels import label_stats, targets_from_manifest
from .text import build_text, describe_mode, is_leaky

LOG = get_logger()

MANIFEST_NAME = "manifest.csv"
CACHE_NAME = "images_u8.npy"
STATS_NAME = "stats.json"


def manifest_path(cfg: DotDict) -> Path:
    return Path(cfg.paths.manifest) / MANIFEST_NAME


def cache_path(cfg: DotDict) -> Path:
    return Path(cfg.paths.cache) / CACHE_NAME


def preprocess_image_pil(image, size: int) -> np.ndarray:
    """Grayscale, shorter-side resize, centre crop -> uint8 [size, size].

    The single definition of "how a film becomes a tensor". `serve.inference`
    calls this too, so the training and serving pipelines cannot drift apart.
    """
    image = image.convert("L")
    width, height = image.size
    if width == 0 or height == 0:
        raise ValueError("image has zero extent")
    scale = size / min(width, height)
    new_size = (max(int(round(width * scale)), size), max(int(round(height * scale)), size))
    image = image.resize(new_size, Image.BILINEAR)
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    image = image.crop((left, top, left + size, top + size))
    return np.asarray(image, dtype=np.uint8)


def preprocess_image(path: str | os.PathLike, size: int) -> Optional[np.ndarray]:
    """File on disk -> cached uint8 array, or None if it cannot be read."""
    try:
        with Image.open(path) as handle:
            return preprocess_image_pil(handle, size)
    except Exception as exc:  # noqa: BLE001 - a corrupt file must not stop the run
        LOG.warning("could not read %s (%s)", path, exc)
        return None


def build_image_cache(
    manifest: pd.DataFrame, raw_dir: Path, out_path: Path, size: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Write the uint8 memmap; drop manifest rows whose image is unreadable."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(manifest)
    memmap = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(n, size, size))

    try:
        from tqdm.auto import tqdm

        iterator = tqdm(enumerate(manifest["rel_path"]), total=n, desc="cache", unit="img")
    except ImportError:
        iterator = enumerate(manifest["rel_path"])

    keep = np.zeros(n, dtype=bool)
    written = 0
    for i, rel in iterator:
        array = preprocess_image(raw_dir / rel, size)
        if array is None:
            continue
        memmap[written] = array
        keep[i] = True
        written += 1
    memmap.flush()
    del memmap

    if written < n:
        LOG.warning("%d/%d images unreadable or missing - dropped from the manifest", n - written, n)
        # Truncate the file to the rows we actually wrote.
        full = np.load(out_path, mmap_mode="r")
        trimmed = np.array(full[:written])
        del full
        np.save(out_path, trimmed)
        manifest = manifest[keep].reset_index(drop=True)
    if written == 0:
        raise RuntimeError(
            f"no images could be read under {raw_dir}. Check that the download step "
            "completed and that manifest rel_paths match the files on disk."
        )

    manifest = manifest.copy()
    manifest["cache_index"] = np.arange(len(manifest), dtype=np.int64)
    return manifest, np.load(out_path, mmap_mode="r")


def prepare(
    cfg: DotDict,
    physionet_user: Optional[str] = None,
    physionet_pass: Optional[str] = None,
    download: bool = True,
    rebuild_cache: bool = True,
) -> pd.DataFrame:
    """Run stage 0 end to end for whichever source the config names."""
    ensure_dirs(cfg)
    raw = Path(cfg.paths.raw)
    source = cfg.data.source

    if source == "mimic":
        tables_present = all((raw / name).exists() for name in mimic.TABLES.values())
        if download and not tables_present:
            mimic.download_tables(raw, physionet_user, physionet_pass)
        manifest = mimic.build_manifest(
            raw_dir=raw,
            subset_size=int(cfg.data.subset_size),
            views=cfg.data.views,
            one_image_per_study=bool(cfg.data.one_image_per_study),
            balance=bool(cfg.data.balance_subset),
            uncertain_policy=cfg.data.uncertain_policy,
            per_class_policy=dict(cfg.data.get("per_class_policy", {}) or {}),
            blank_policy=cfg.data.get("blank_policy", "zeros"),
            drop_unlabeled_rows=bool(cfg.data.drop_unlabeled_rows),
            seed=int(cfg.seed),
        )
        if download:
            mimic.download_images(manifest, raw, physionet_user, physionet_pass)
        reports = mimic.fetch_reports(manifest, raw, physionet_user, physionet_pass)
        manifest["report_text"] = [reports.get(rel, "") for rel in manifest["report_rel_path"]]
        patients_csv = cfg.data.get("patients_csv", "")
        if patients_csv and Path(patients_csv).exists():
            manifest = mimic.attach_demographics(manifest, patients_csv)

    elif source == "openi":
        if download:
            openi.download(raw, what="both")
        manifest = openi.build_manifest(
            raw_dir=raw,
            subset_size=int(cfg.data.subset_size),
            val_fraction=float(cfg.data.val_fraction),
            seed=int(cfg.seed),
            uncertain_policy=cfg.data.uncertain_policy,
            per_class_policy=dict(cfg.data.get("per_class_policy", {}) or {}),
        )

    elif source == "synthetic":
        n = int(cfg.data.subset_size) or 600
        manifest = synthetic.build_dataset(
            raw_dir=raw, n=n, size=int(cfg.data.cache_size), seed=int(cfg.seed)
        )

    else:
        raise ValueError(f"unknown data.source '{source}' (mimic | openi | synthetic)")

    # ---- text channel ------------------------------------------------------
    mode = cfg.text.mode
    manifest["text"] = [
        build_text(report, mode=mode, lowercase=bool(cfg.text.lowercase))
        for report in manifest.get("report_text", [""] * len(manifest))
    ]
    empty_text = int((manifest["text"].str.len() == 0).sum())
    manifest.loc[manifest["text"].str.len() == 0, "text"] = cfg.text.empty_placeholder
    LOG.info(
        "text mode '%s' (%s); %d/%d rows fell back to the placeholder",
        mode, describe_mode(mode), empty_text, len(manifest),
    )
    if is_leaky(mode):
        LOG.warning(
            "text.mode='%s' contains the sentences the CheXpert labels were derived "
            "from. Results are an upper bound, not a clinical estimate. See docs/LEAKAGE.md.",
            mode,
        )

    # ---- image cache -------------------------------------------------------
    cache_file = cache_path(cfg)
    if rebuild_cache or not cache_file.exists():
        manifest, _ = build_image_cache(manifest, raw, cache_file, int(cfg.data.cache_size))
    else:
        LOG.info("reusing existing image cache at %s", cache_file)
        manifest["cache_index"] = np.arange(len(manifest), dtype=np.int64)

    # ---- persist -----------------------------------------------------------
    out = manifest_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = [
        c for c in manifest.columns if c not in ("report_text",)
    ]  # raw reports are large and may contain PHI-adjacent text; keep them out
    manifest[keep_cols].to_csv(out, index=False)
    LOG.info("wrote %s (%d rows)", out, len(manifest))

    y, mask = targets_from_manifest(manifest)
    stats: Dict = {
        "source": source,
        "rows": int(len(manifest)),
        "image_size": int(cfg.data.image_size),
        "cache_size": int(cfg.data.cache_size),
        "splits": {k: int(v) for k, v in manifest["split"].value_counts().items()},
        "text_mode": mode,
        "text_mode_description": describe_mode(mode),
        "text_is_leaky": is_leaky(mode),
        "uncertain_policy": cfg.data.uncertain_policy,
        "blank_policy": cfg.data.get("blank_policy", "zeros"),
        "labels": LABELS,
        "label_stats_all": label_stats(y, mask),
    }
    for split_name in ("train", "val", "test"):
        rows = manifest["split"] == split_name
        if rows.any():
            stats[f"label_stats_{split_name}"] = label_stats(y[rows.to_numpy()], mask[rows.to_numpy()])
    save_json(stats, Path(cfg.paths.manifest) / STATS_NAME)

    _sanity_check(manifest)
    return manifest


def _sanity_check(manifest: pd.DataFrame) -> None:
    """Fail loudly on the mistakes that silently ruin a run."""
    problems = []
    if "split" not in manifest.columns:
        problems.append("manifest has no split column")
    else:
        by_split = manifest.groupby("split")["subject_id"].apply(set)
        names = list(by_split.index)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                shared = by_split.iloc[i] & by_split.iloc[j]
                if shared:
                    problems.append(
                        f"{len(shared)} patients appear in both '{names[i]}' and '{names[j]}' - "
                        "this leaks and will inflate every metric"
                    )
    for split_name in ("train", "val", "test"):
        if (manifest["split"] == split_name).sum() == 0:
            problems.append(f"split '{split_name}' is empty")
    y, mask = targets_from_manifest(manifest)
    for c, name in enumerate(LABELS):
        for split_name in ("train", "val", "test"):
            rows = (manifest["split"] == split_name).to_numpy()
            if rows.sum() == 0:
                continue
            valid = mask[rows, c] > 0
            positives = y[rows, c][valid].sum()
            if valid.sum() and positives == 0:
                problems.append(f"'{name}' has no positive case in split '{split_name}' — AUC undefined")

    if problems:
        for problem in problems:
            LOG.error("SANITY: %s", problem)
        raise RuntimeError("manifest failed sanity checks:\n  - " + "\n  - ".join(problems))
    LOG.info("manifest sanity checks passed (no patient overlap, every class present in every split)")


def load_manifest(cfg: DotDict) -> pd.DataFrame:
    path = manifest_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `dvlhg prep` first")
    frame = pd.read_csv(path)
    frame["text"] = frame["text"].fillna("").astype(str)
    return frame


def load_image_cache(cfg: DotDict) -> np.ndarray:
    path = cache_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `dvlhg prep` first")
    return np.load(path, mmap_mode="r")
