"""MIMIC-CXR-JPG v2.1.0 -> project manifest.

PhysioNet credentialed access is required (CITI training + a signed DUA). The
full JPG release is ~570 GB, which no Colab session will hold, so the flow here
is deliberately subset-first:

  1. download four small metadata tables (a few MB total)
  2. build a manifest of N studies, stratified over the label combinations
  3. download only those N JPEGs and their reports

Nothing outside the manifest is ever fetched.
"""

from __future__ import annotations

import concurrent.futures as futures
import gzip
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..constants import FRONTAL_VIEWS, LABELS
from ..utils import get_logger
from .labels import attach_targets, combo_key, targets_from_manifest

LOG = get_logger()

JPG_BASE = "https://physionet.org/files/mimic-cxr-jpg/2.1.0"
REPORT_BASE = "https://physionet.org/files/mimic-cxr/2.0.0"
REPORTS_ZIP = f"{REPORT_BASE}/mimic-cxr-reports.zip"

TABLES = {
    "chexpert": "mimic-cxr-2.0.0-chexpert.csv.gz",
    "metadata": "mimic-cxr-2.0.0-metadata.csv.gz",
    "split": "mimic-cxr-2.0.0-split.csv.gz",
}

# View ranking used when a study has several frontal films: prefer PA.
VIEW_RANK = {"PA": 0, "AP": 1, "AP AXIAL": 2, "LATERAL": 8, "LL": 9}


# --------------------------------------------------------------------------- #
# download helpers
# --------------------------------------------------------------------------- #
def _session(user: str, password: str):
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError as exc:  # pragma: no cover
        raise ImportError("`pip install requests` is required to download MIMIC") from exc

    sess = requests.Session()
    sess.auth = (user, password)
    retry = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=32, pool_connections=32)
    sess.mount("https://", adapter)
    return sess


def _credentials(user: Optional[str], password: Optional[str]) -> Tuple[str, str]:
    user = user or os.environ.get("PHYSIONET_USER", "")
    password = password or os.environ.get("PHYSIONET_PASS", "")
    if not user or not password:
        raise RuntimeError(
            "PhysioNet credentials missing. Set PHYSIONET_USER / PHYSIONET_PASS, or "
            "pass --user/--password. You need credentialed access to MIMIC-CXR-JPG."
        )
    return user, password


def _fetch(sess, url: str, dest: Path, overwrite: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with sess.get(url, stream=True, timeout=120) as response:
        if response.status_code == 401:
            raise RuntimeError(
                "PhysioNet returned 401. Check your username/password and that your "
                "account has signed the MIMIC-CXR-JPG data use agreement."
            )
        response.raise_for_status()
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    os.replace(tmp, dest)
    return dest


def download_tables(
    raw_dir: str | os.PathLike,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> Dict[str, Path]:
    """Fetch the three small metadata tables (~40 MB total)."""
    user, password = _credentials(user, password)
    sess = _session(user, password)
    raw = Path(raw_dir)
    out: Dict[str, Path] = {}
    for key, name in TABLES.items():
        LOG.info("downloading %s", name)
        out[key] = _fetch(sess, f"{JPG_BASE}/{name}", raw / name)
    return out


def download_reports_zip(
    raw_dir: str | os.PathLike,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> Optional[Path]:
    """Fetch mimic-cxr-reports.zip (~130 MB) from the MIMIC-CXR v2.0.0 project.

    Returns None if the archive is unavailable — the per-study fallback in
    `fetch_reports` handles that case.
    """
    user, password = _credentials(user, password)
    sess = _session(user, password)
    dest = Path(raw_dir) / "mimic-cxr-reports.zip"
    try:
        LOG.info("downloading mimic-cxr-reports.zip (~130 MB, one time)")
        return _fetch(sess, REPORTS_ZIP, dest)
    except Exception as exc:  # noqa: BLE001 - any failure falls back
        LOG.warning("reports zip unavailable (%s); will fetch reports per study", exc)
        if dest.exists():
            dest.unlink()
        return None


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def _read_table(path: Path) -> pd.DataFrame:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as handle:
            return pd.read_csv(handle)
    return pd.read_csv(path)


def image_rel_path(subject_id: int, study_id: int, dicom_id: str) -> str:
    """files/p10/p10000032/s50414267/<dicom>.jpg"""
    subject = str(int(subject_id))
    return f"files/p{subject[:2]}/p{subject}/s{int(study_id)}/{dicom_id}.jpg"


def report_rel_path(subject_id: int, study_id: int) -> str:
    """files/p10/p10000032/s50414267.txt"""
    subject = str(int(subject_id))
    return f"files/p{subject[:2]}/p{subject}/s{int(study_id)}.txt"


def build_manifest(
    raw_dir: str | os.PathLike,
    subset_size: int = 12000,
    views: str = "frontal",
    one_image_per_study: bool = True,
    balance: bool = True,
    uncertain_policy: str = "ignore",
    per_class_policy: Optional[Dict[str, str]] = None,
    blank_policy: str = "zeros",
    drop_unlabeled_rows: bool = True,
    seed: int = 1337,
) -> pd.DataFrame:
    """Join labels + metadata + official split, then take a stratified subset."""
    raw = Path(raw_dir)
    for key, name in TABLES.items():
        if not (raw / name).exists():
            raise FileNotFoundError(
                f"{raw / name} not found. Run `dvlhg prep --download-tables` first."
            )

    chex = _read_table(raw / TABLES["chexpert"])
    meta = _read_table(raw / TABLES["metadata"])
    split = _read_table(raw / TABLES["split"])

    keep_meta = ["dicom_id", "subject_id", "study_id", "ViewPosition"]
    for extra in ("StudyDate", "PerformedProcedureStepDescription"):
        if extra in meta.columns:
            keep_meta.append(extra)
    meta = meta[keep_meta]

    frame = meta.merge(split[["dicom_id", "split"]], on="dicom_id", how="inner")
    frame = frame.merge(
        chex[["subject_id", "study_id"] + LABELS],
        on=["subject_id", "study_id"],
        how="inner",
    )
    LOG.info("joined tables: %d images with labels and a split", len(frame))

    frame["ViewPosition"] = frame["ViewPosition"].fillna("UNKNOWN").astype(str).str.upper()
    if views == "frontal":
        frame = frame[frame["ViewPosition"].isin(FRONTAL_VIEWS)].copy()
        LOG.info("frontal views only: %d images", len(frame))

    if one_image_per_study:
        frame["_rank"] = frame["ViewPosition"].map(VIEW_RANK).fillna(5).astype(int)
        frame = (
            frame.sort_values(["subject_id", "study_id", "_rank", "dicom_id"])
            .groupby(["subject_id", "study_id"], as_index=False)
            .first()
            .drop(columns=["_rank"])
        )
        LOG.info("one image per study: %d studies", len(frame))

    if drop_unlabeled_rows:
        mentioned = frame[LABELS].notna().any(axis=1)
        frame = frame[mentioned].copy()
        LOG.info("studies mentioning >=1 target finding: %d", len(frame))

    frame = attach_targets(frame, uncertain_policy, per_class_policy, blank_policy)
    # MIMIC's official split calls the validation fold "validate".
    frame["split"] = frame["split"].replace({"validate": "val"})
    frame["rel_path"] = [
        image_rel_path(s, st, d)
        for s, st, d in zip(frame["subject_id"], frame["study_id"], frame["dicom_id"])
    ]
    frame["report_rel_path"] = [
        report_rel_path(s, st) for s, st in zip(frame["subject_id"], frame["study_id"])
    ]
    frame["source"] = "mimic"

    if subset_size and subset_size < len(frame):
        frame = stratified_subset(frame, subset_size, balance=balance, seed=seed)

    return frame.reset_index(drop=True)


def stratified_subset(
    frame: pd.DataFrame, size: int, balance: bool = True, seed: int = 1337
) -> pd.DataFrame:
    """Subset while (a) honouring the official split proportions and (b) keeping
    rare label combinations alive.

    Patients never straddle splits here because the official MIMIC split is
    already patient-level and we only ever sample *within* a split.
    """
    rng = np.random.default_rng(seed)
    y, _ = targets_from_manifest(frame)
    frame = frame.copy()
    frame["_combo"] = [combo_key(row) for row in y]

    pieces: List[pd.DataFrame] = []
    for split_name, group in frame.groupby("split", sort=False):
        quota = max(int(round(size * len(group) / len(frame))), 1)
        if quota >= len(group):
            pieces.append(group)
            continue
        if not balance:
            pieces.append(group.sample(n=quota, random_state=seed))
            continue

        # Square-root allocation: keeps the head dominant but guarantees the
        # tail combinations survive, instead of a hard uniform re-balance that
        # would distort prevalence beyond recognition.
        counts = group["_combo"].value_counts()
        weights = np.sqrt(counts.to_numpy(dtype=np.float64))
        alloc = np.maximum(np.floor(weights / weights.sum() * quota), 1).astype(int)
        alloc = np.minimum(alloc, counts.to_numpy())
        # Hand any remainder to the largest combinations.
        deficit = quota - int(alloc.sum())
        order = np.argsort(-counts.to_numpy())
        idx = 0
        while deficit > 0 and idx < len(order) * 4:
            j = order[idx % len(order)]
            if alloc[j] < counts.iloc[j]:
                alloc[j] += 1
                deficit -= 1
            idx += 1

        chosen: List[pd.DataFrame] = []
        for combo, take in zip(counts.index, alloc):
            rows = group[group["_combo"] == combo]
            take = int(min(take, len(rows)))
            if take > 0:
                chosen.append(rows.sample(n=take, random_state=int(rng.integers(1 << 31))))
        pieces.append(pd.concat(chosen, axis=0))

    out = pd.concat(pieces, axis=0).drop(columns=["_combo"])
    LOG.info(
        "stratified subset: %d rows (%s)",
        len(out),
        ", ".join(f"{k}={v}" for k, v in out["split"].value_counts().items()),
    )
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# bulk fetch of the subset
# --------------------------------------------------------------------------- #
def download_images(
    manifest: pd.DataFrame,
    raw_dir: str | os.PathLike,
    user: Optional[str] = None,
    password: Optional[str] = None,
    workers: int = 8,
    progress: bool = True,
) -> int:
    """Download exactly the JPEGs named in the manifest. Resumable."""
    user, password = _credentials(user, password)
    sess = _session(user, password)
    raw = Path(raw_dir)
    todo = [
        rel for rel in manifest["rel_path"]
        if not (raw / rel).exists() or (raw / rel).stat().st_size == 0
    ]
    LOG.info("%d/%d images already present; fetching %d", len(manifest) - len(todo), len(manifest), len(todo))
    if not todo:
        return 0

    failures: List[str] = []

    def one(rel: str) -> bool:
        try:
            _fetch(sess, f"{JPG_BASE}/{rel}", raw / rel)
            return True
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{rel}: {exc}")
            return False

    done = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        iterator = pool.map(one, todo)
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(iterator, total=len(todo), desc="images", unit="img")
            except ImportError:
                pass
        for ok in iterator:
            done += int(ok)

    if failures:
        LOG.warning("%d image downloads failed; first few: %s", len(failures), failures[:3])
    return done


def fetch_reports(
    manifest: pd.DataFrame,
    raw_dir: str | os.PathLike,
    user: Optional[str] = None,
    password: Optional[str] = None,
    workers: int = 8,
    use_zip: bool = True,
) -> Dict[str, str]:
    """Return {report_rel_path: raw text} for every study in the manifest.

    Prefers the one-shot reports archive; falls back to per-study downloads.
    """
    raw = Path(raw_dir)
    wanted = list(dict.fromkeys(manifest["report_rel_path"]))
    texts: Dict[str, str] = {}

    zip_path = raw / "mimic-cxr-reports.zip"
    if use_zip and not zip_path.exists():
        download_reports_zip(raw, user, password)
    if use_zip and zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
            for rel in wanted:
                for candidate in (rel, f"mimic-cxr-reports/{rel}", rel.lstrip("/")):
                    if candidate in names:
                        texts[rel] = archive.read(candidate).decode("utf-8", errors="replace")
                        break
        LOG.info("read %d/%d reports from the archive", len(texts), len(wanted))

    missing = [rel for rel in wanted if rel not in texts]
    for rel in list(missing):
        local = raw / rel
        if local.exists():
            texts[rel] = local.read_text(encoding="utf-8", errors="replace")
            missing.remove(rel)

    if missing:
        user, password = _credentials(user, password)
        sess = _session(user, password)
        LOG.info("fetching %d reports individually", len(missing))

        def one(rel: str) -> Tuple[str, str]:
            try:
                path = _fetch(sess, f"{REPORT_BASE}/{rel}", raw / rel)
                return rel, path.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                return rel, ""

        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for rel, text in pool.map(one, missing):
                texts[rel] = text

    empty = sum(1 for value in texts.values() if not value.strip())
    if empty:
        LOG.warning("%d reports came back empty", empty)
    return texts


def attach_demographics(manifest: pd.DataFrame, patients_csv: str | os.PathLike) -> pd.DataFrame:
    """Optional: join MIMIC-IV `patients.csv` for the demographic hyperedges.

    MIMIC-CXR-JPG itself carries no age or sex. Without this file the
    demographic hyperedge group is skipped automatically.
    """
    patients = pd.read_csv(patients_csv)
    cols = {c.lower(): c for c in patients.columns}
    need = {"subject_id", "gender", "anchor_age"}
    if not need.issubset(cols):
        raise KeyError(f"{patients_csv} must contain {sorted(need)}; found {list(patients.columns)}")
    patients = patients[[cols["subject_id"], cols["gender"], cols["anchor_age"]]]
    patients.columns = ["subject_id", "sex", "age"]
    return manifest.merge(patients, on="subject_id", how="left")
