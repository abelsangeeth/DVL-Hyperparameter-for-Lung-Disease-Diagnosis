"""Open-i / Indiana University Chest X-ray -> project manifest.

A public, no-credentials fallback for anyone who does not (yet) have PhysioNet
access to MIMIC-CXR. ~7.4k images, ~3.9k reports, each with real FINDINGS /
IMPRESSION / INDICATION sections and MeSH annotations.

Caveats, stated up front because they change how results should be read:
  * Labels here are derived from MeSH terms, not from the CheXpert labeler.
    There is no "uncertain" state — a finding is either annotated or absent.
  * The XML carries no view position. We keep the first `parentImage` of each
    report, which is the frontal film in the large majority of studies.
  * There is no official split, so we make a reproducible study-level one.
"""

from __future__ import annotations

import os
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils import get_logger
from .labels import attach_targets

LOG = get_logger()

IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"

# MeSH surface forms -> our four labels. Matching is case-insensitive and
# substring-based because Open-i qualifies terms, e.g.
# "Pulmonary Atelectasis/base/left" or "Pleural Effusion/bilateral".
MESH_MAP: Dict[str, List[str]] = {
    "Atelectasis": ["atelectasis"],
    "Cardiomegaly": ["cardiomegaly", "cardiac enlargement", "hypertrophy, left ventricular"],
    "Edema": ["pulmonary edema", "edema, pulmonary", "pulmonary congestion", "hypertension, pulmonary"],
    "Pleural Effusion": ["pleural effusion", "hydrothorax", "pleural fluid"],
}


def download(raw_dir: str | os.PathLike, what: str = "both") -> None:
    """Fetch and unpack the Open-i archives (no credentials needed)."""
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise ImportError("`pip install requests` is required") from exc

    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    jobs = []
    if what in ("both", "reports"):
        jobs.append((REPORTS_URL, raw / "NLMCXR_reports.tgz", raw / "reports"))
    if what in ("both", "images"):
        jobs.append((IMAGES_URL, raw / "NLMCXR_png.tgz", raw / "images"))

    for url, archive, target in jobs:
        if target.exists() and any(target.rglob("*")):
            LOG.info("%s already unpacked", target.name)
            continue
        if not archive.exists():
            LOG.info("downloading %s (this one is large for images: ~3.9 GB)", url)
            with requests.get(url, stream=True, timeout=300) as response:
                response.raise_for_status()
                tmp = archive.with_suffix(".part")
                with open(tmp, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
                os.replace(tmp, archive)
        LOG.info("unpacking %s", archive.name)
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            _safe_extract(tar, target)


def _safe_extract(tar: tarfile.TarFile, target: Path) -> None:
    """Reject entries that would escape the target directory."""
    target = target.resolve()
    for member in tar.getmembers():
        dest = (target / member.name).resolve()
        if not str(dest).startswith(str(target)):
            raise RuntimeError(f"refusing unsafe tar entry: {member.name}")
    # filter="data" exists on Python 3.12+; older versions fall back silently.
    try:
        tar.extractall(target, filter="data")  # type: ignore[call-arg]
    except TypeError:
        tar.extractall(target)


def _section(root: ET.Element, label: str) -> str:
    for node in root.iter("AbstractText"):
        if (node.get("Label") or "").strip().upper() == label:
            return (node.text or "").strip()
    return ""


def _mesh_terms(root: ET.Element) -> List[str]:
    terms: List[str] = []
    for mesh in root.iter("MeSH"):
        for child in mesh:
            if child.tag in ("major", "minor", "automatic") and child.text:
                terms.append(child.text.strip())
    return terms


def parse_reports(reports_dir: str | os.PathLike) -> pd.DataFrame:
    """Parse every ecgen-radiology XML into one row per report."""
    root_dir = Path(reports_dir)
    files = sorted(root_dir.rglob("*.xml"))
    if not files:
        raise FileNotFoundError(f"no XML reports under {root_dir}")

    rows: List[dict] = []
    for path in files:
        try:
            tree = ET.parse(path)
        except ET.ParseError:
            continue
        root = tree.getroot()
        images = [node.get("id") for node in root.iter("parentImage") if node.get("id")]
        if not images:
            continue
        terms = _mesh_terms(root)
        joined = " | ".join(terms).lower()
        record = {
            "uid": path.stem,
            "image_ids": ";".join(images),
            "dicom_id": images[0],
            "mesh": " | ".join(terms),
            "comparison": _section(root, "COMPARISON"),
            "indication": _section(root, "INDICATION"),
            "findings": _section(root, "FINDINGS"),
            "impression": _section(root, "IMPRESSION"),
        }
        for name, needles in MESH_MAP.items():
            record[name] = 1.0 if any(needle in joined for needle in needles) else 0.0
        rows.append(record)

    frame = pd.DataFrame(rows)
    LOG.info("parsed %d Open-i reports", len(frame))
    return frame


def to_report_text(row: pd.Series) -> str:
    """Rebuild a MIMIC-shaped report so the same section parser works."""
    parts = []
    if row.get("comparison"):
        parts.append(f"COMPARISON: {row['comparison']}")
    if row.get("indication"):
        parts.append(f"INDICATION: {row['indication']}")
    if row.get("findings"):
        parts.append(f"FINDINGS: {row['findings']}")
    if row.get("impression"):
        parts.append(f"IMPRESSION: {row['impression']}")
    return "\n\n".join(parts)


def build_manifest(
    raw_dir: str | os.PathLike,
    subset_size: int = 0,
    val_fraction: float = 0.10,
    test_fraction: float = 0.20,
    seed: int = 1337,
    uncertain_policy: str = "ignore",
    per_class_policy: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    raw = Path(raw_dir)
    frame = parse_reports(raw / "reports")

    images_dir = raw / "images"
    available = {path.stem: path for path in images_dir.rglob("*.png")}
    if available:
        frame = frame[frame["dicom_id"].isin(available)].copy()
        frame["rel_path"] = [
            str(available[uid].relative_to(raw)).replace("\\", "/") for uid in frame["dicom_id"]
        ]
        LOG.info("%d reports have their frontal image on disk", len(frame))
    else:
        LOG.warning("no PNGs found under %s - manifest will point at expected paths", images_dir)
        frame["rel_path"] = [f"images/{uid}.png" for uid in frame["dicom_id"]]

    # Open-i MeSH labels carry no uncertainty, so the policy is a no-op here;
    # it is applied anyway so the y_/m_ columns exist in the same shape.
    frame = attach_targets(frame, uncertain_policy, per_class_policy, blank_policy="zeros")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    n_test = int(round(len(frame) * test_fraction))
    n_val = int(round(len(frame) * val_fraction))
    split = np.array(["train"] * len(frame), dtype=object)
    split[order[:n_test]] = "test"
    split[order[n_test : n_test + n_val]] = "val"
    frame = frame.assign(split=split)

    frame["subject_id"] = frame["uid"]
    frame["study_id"] = frame["uid"]
    frame["ViewPosition"] = "PA"  # not recorded by Open-i; see module docstring
    frame["report_rel_path"] = ""
    frame["report_text"] = [to_report_text(row) for _, row in frame.iterrows()]
    frame["source"] = "openi"

    if subset_size and subset_size < len(frame):
        frame = frame.sample(n=subset_size, random_state=seed)

    LOG.info(
        "Open-i manifest: %d rows (%s)",
        len(frame),
        ", ".join(f"{k}={v}" for k, v in frame["split"].value_counts().items()),
    )
    return frame.reset_index(drop=True)
