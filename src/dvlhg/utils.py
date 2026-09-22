"""Seeding, devices, logging, checkpoint helpers. Shared by every stage."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

_LOG_READY = False


def get_logger(name: str = "dvlhg", logfile: str | os.PathLike | None = None) -> logging.Logger:
    global _LOG_READY
    logger = logging.getLogger(name)
    if not _LOG_READY:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        root = logging.getLogger("dvlhg")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.propagate = False
        _LOG_READY = True
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        already = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == Path(logfile).resolve()
            for h in logging.getLogger("dvlhg").handlers
        )
        if not already:
            fh = logging.FileHandler(logfile, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
            logging.getLogger("dvlhg").addHandler(fh)
    return logger


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def pick_device(spec: str = "auto") -> torch.device:
    if spec and spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_enabled(cfg_flag: bool, device: torch.device) -> bool:
    """AMP only ever helps on CUDA here; on CPU it is slower and noisier."""
    return bool(cfg_flag) and device.type == "cuda"


def count_params(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def human(n: float) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n:.0f}"
        n /= 1000.0
    return f"{n:.1f}T"


@contextmanager
def timed(label: str, logger: logging.Logger | None = None):
    start = time.perf_counter()
    yield
    msg = f"{label} took {time.perf_counter() - start:.1f}s"
    (logger or get_logger()).info(msg)


def save_json(obj: Any, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def save_checkpoint(state: dict, path: str | os.PathLike) -> None:
    """Atomic save - a Colab disconnect mid-write must not corrupt the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | os.PathLike, map_location: Any = "cpu") -> dict:
    # weights_only=False: our checkpoints carry config dicts and numpy arrays.
    # Only ever load checkpoints you produced yourself.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only kwarg
        return torch.load(path, map_location=map_location)


class EMA:
    """Exponential moving average of the trainable parameters."""

    def __init__(self, module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone().float()
            for name, param in module.named_parameters()
            if param.requires_grad
        }
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        for name, param in module.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, module: torch.nn.Module) -> None:
        self._backup = {}
        for name, param in module.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.detach().clone()
                param.copy_(self.shadow[name].to(param.dtype))

    @torch.no_grad()
    def restore(self, module: torch.nn.Module) -> None:
        for name, param in module.named_parameters():
            if name in self._backup:
                param.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def cosine_warmup_lambda(total_steps: int, warmup_steps: int, min_scale: float = 0.02):
    """LR multiplier: linear warm-up then cosine decay to `min_scale`."""
    total_steps = max(total_steps, 1)
    warmup_steps = max(min(warmup_steps, total_steps - 1), 0)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_scale + (1 - min_scale) * 0.5 * (1 + np.cos(np.pi * progress))

    return fn
