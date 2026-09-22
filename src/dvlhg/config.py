"""Tiny config system: YAML file + dotted-key overrides, no extra dependencies.

    cfg = load_config("configs/default.yaml", ["train.lr=1e-4", "data.subset_size=4000"])
    cfg.train.lr        -> 0.0001
    cfg["train"]["lr"]  -> 0.0001

Every stage writes the exact config it ran with next to its checkpoint, so a
result can always be traced back to the settings that produced it.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class DotDict(dict):
    """dict whose keys are also attributes, recursively."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, DotDict):
            return value
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - attribute protocol
            raise AttributeError(
                f"config has no key '{name}'. Available: {sorted(self.keys())}"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def __delattr__(self, name: str) -> None:
        del self[name]

    def to_dict(self) -> dict:
        out = {}
        for key, value in self.items():
            if isinstance(value, DotDict):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [v.to_dict() if isinstance(v, DotDict) else v for v in value]
            else:
                out[key] = value
        return out

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], Mapping):
                node[part] = DotDict()
            node = node[part]
        node[parts[-1]] = DotDict._wrap(value)

    def copy(self) -> "DotDict":  # type: ignore[override]
        return DotDict(copy.deepcopy(self.to_dict()))


def _coerce(text: str) -> Any:
    """Turn a CLI string into the obvious Python value."""
    lowered = text.strip().lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"none", "null", "~"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    stripped = text.strip()
    if stripped.startswith(("[", "{")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    if "," in stripped:
        return [_coerce(part) for part in stripped.split(",")]
    return text


def _deep_merge(base: dict, extra: Mapping) -> dict:
    for key, value in extra.items():
        if key in base and isinstance(base[key], Mapping) and isinstance(value, Mapping):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


def default_config_path() -> Path:
    """configs/default.yaml, found from the installed package or the repo."""
    env = os.environ.get("DVLHG_CONFIG")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs" / "default.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "configs/default.yaml not found; pass --config or set DVLHG_CONFIG"
    )


def load_config(
    path: str | os.PathLike | None = None,
    overrides: Iterable[str] | None = None,
) -> DotDict:
    """Load YAML config, apply `key.sub=value` overrides, resolve paths."""
    path = Path(path) if path else default_config_path()
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    # A config may inherit from another with `_base_: other.yaml`.
    base_name = raw.pop("_base_", None)
    if base_name:
        base = load_config(Path(path).parent / base_name).to_dict()
        raw = _deep_merge(base, raw)

    cfg = DotDict(raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like key.sub=value")
        key, _, value = item.partition("=")
        cfg.set_path(key.strip(), _coerce(value))

    cfg.set_path("_config_path", str(Path(path).resolve()))
    return resolve_paths(cfg)


def resolve_paths(cfg: DotDict) -> DotDict:
    """Expand ~ and $VARS, and make every `paths.*` entry absolute."""
    root = Path(os.path.expandvars(os.path.expanduser(str(cfg.paths.root)))).resolve()
    cfg.paths.root = str(root)
    for key, value in list(cfg.paths.items()):
        if key == "root" or not isinstance(value, str):
            continue
        expanded = Path(os.path.expandvars(os.path.expanduser(value)))
        cfg.paths[key] = str(expanded if expanded.is_absolute() else root / expanded)
    return cfg


def save_config(cfg: DotDict, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False, default_flow_style=False)


def ensure_dirs(cfg: DotDict) -> None:
    """Create every directory named in `paths` (files are skipped)."""
    for key, value in cfg.paths.items():
        if key.startswith("_") or not isinstance(value, str):
            continue
        if Path(value).suffix:  # looks like a file, not a directory
            Path(value).parent.mkdir(parents=True, exist_ok=True)
        else:
            Path(value).mkdir(parents=True, exist_ok=True)
