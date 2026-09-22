"""Bundle the code for upload to Colab (when you would rather not use GitHub).

    python scripts/make_colab_zip.py

Writes dist/dvl-hypergraph.zip containing only source, configs, notebooks,
frontend, docs and tests. Run directories, checkpoints, caches and any
downloaded data are excluded — the archive should be a couple of hundred
kilobytes, and if it is much larger something unwanted got in.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
ARCHIVE = DIST / "dvl-hypergraph.zip"

INCLUDE_DIRS = ["src", "configs", "notebooks", "frontend", "scripts", "docs", "tests"]
INCLUDE_FILES = ["README.md", "PLAN.md", "requirements.txt", "pyproject.toml", ".gitignore"]

SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", "runs", "dist", ".git"}
SKIP_SUFFIXES = {".pt", ".npy", ".npz", ".pyc", ".log", ".gz", ".zip", ".tgz", ".tar"}


def keep(path: Path) -> bool:
    if any(part in SKIP_PARTS for part in path.parts):
        return False
    return path.suffix.lower() not in SKIP_SUFFIXES


def main() -> None:
    DIST.mkdir(exist_ok=True)
    added = 0
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in INCLUDE_FILES:
            path = ROOT / name
            if path.exists():
                archive.write(path, f"dvl-hypergraph/{name}")
                added += 1
        for directory in INCLUDE_DIRS:
            base = ROOT / directory
            if not base.exists():
                continue
            for path in sorted(base.rglob("*")):
                if path.is_file() and keep(path.relative_to(ROOT)):
                    archive.write(path, f"dvl-hypergraph/{path.relative_to(ROOT).as_posix()}")
                    added += 1

    size_kb = ARCHIVE.stat().st_size / 1024
    print(f"{ARCHIVE}  —  {added} files, {size_kb:.0f} KB")
    if size_kb > 5000:
        print("WARNING: larger than 5 MB. Something big slipped past the filters.")
    print("\nUpload this in notebook cell 2 with SOURCE = 'zip'.")


if __name__ == "__main__":
    main()
