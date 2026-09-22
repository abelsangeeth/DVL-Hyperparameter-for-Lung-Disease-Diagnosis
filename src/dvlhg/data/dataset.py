"""Torch dataset over the cached films + tokenised reports.

Notes that matter clinically:
  * No horizontal flip. Chest anatomy is not left/right symmetric — flipping a
    film turns a normal heart into dextrocardia and destroys the very cue the
    Cardiomegaly head needs.
  * Augmentation is mild (small crop / rotation / photometric jitter). Aggressive
    geometric augmentation erases the costophrenic angle that Pleural Effusion
    depends on.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .labels import targets_from_manifest

TokenizeFn = Callable[[Sequence[str]], Dict[str, torch.Tensor]]


class CXRDataset(Dataset):
    """One row = one study: film, report tokens, targets, loss mask."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        images: Dict[str, np.ndarray],
        tokenize: Optional[TokenizeFn],
        image_size: int = 224,
        train: bool = False,
        modality_dropout: float = 0.0,
        null_text: str = "",
        max_length: int = 256,
        synthetic_weight: float = 0.5,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.images = images
        self.image_size = int(image_size)
        self.train = bool(train)
        self.modality_dropout = float(modality_dropout)

        y, mask = targets_from_manifest(self.manifest)
        self.y = torch.from_numpy(y)
        self.mask = torch.from_numpy(mask)

        self.cache_index = self.manifest["cache_index"].to_numpy(dtype=np.int64)
        if "cache_source" in self.manifest.columns:
            self.cache_source = self.manifest["cache_source"].astype(str).to_numpy()
        else:
            self.cache_source = np.array(["real"] * len(self.manifest), dtype=object)
        self.is_synth = torch.from_numpy((self.cache_source == "synth").astype(np.float32))
        self.weight = torch.where(
            self.is_synth > 0, torch.tensor(float(synthetic_weight)), torch.tensor(1.0)
        )

        texts: List[str] = self.manifest["text"].fillna("").astype(str).tolist()
        if tokenize is not None:
            tokens = tokenize(texts)
            self.input_ids = tokens["input_ids"]
            self.attention_mask = tokens["attention_mask"]
            null_tokens = tokenize([null_text])
            self.null_ids = null_tokens["input_ids"][0]
            self.null_mask = null_tokens["attention_mask"][0]
        else:  # pooled-feature / image-only paths do not need tokens
            self.input_ids = torch.zeros((len(texts), max_length), dtype=torch.long)
            self.attention_mask = torch.zeros((len(texts), max_length), dtype=torch.long)
            self.null_ids = self.input_ids[0]
            self.null_mask = self.attention_mask[0]

    def __len__(self) -> int:
        return len(self.manifest)

    # -- image pipeline ----------------------------------------------------- #
    def _load_image(self, idx: int) -> torch.Tensor:
        source = self.cache_source[idx]
        array = self.images[source][self.cache_index[idx]]
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float().div_(255.0)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor  # [1, S, S] in [0, 1]

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        size = self.image_size
        _, height, width = image.shape

        if not self.train:
            return _center_crop_resize(image, size)

        # Mild random resized crop: 85-100% of the frame, aspect 0.95-1.05.
        scale = float(np.random.uniform(0.85, 1.0))
        ratio = float(np.random.uniform(0.95, 1.05))
        crop_h = int(round(height * np.sqrt(scale / ratio)))
        crop_w = int(round(width * np.sqrt(scale * ratio)))
        crop_h = max(8, min(crop_h, height))
        crop_w = max(8, min(crop_w, width))
        top = int(np.random.randint(0, height - crop_h + 1))
        left = int(np.random.randint(0, width - crop_w + 1))
        image = image[:, top : top + crop_h, left : left + crop_w]

        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(0)

        if np.random.rand() < 0.5:  # small rotation, ±7 degrees
            image = _rotate(image, float(np.random.uniform(-7, 7)))

        if np.random.rand() < 0.8:  # brightness / contrast
            brightness = float(np.random.uniform(-0.08, 0.08))
            contrast = float(np.random.uniform(0.88, 1.12))
            image = ((image - 0.5) * contrast + 0.5 + brightness).clamp_(0, 1)

        return image

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self._augment(self._load_image(idx))

        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        if self.train and self.modality_dropout > 0 and np.random.rand() < self.modality_dropout:
            input_ids, attention_mask = self.null_ids, self.null_mask

        return {
            "image": image,                       # [1, H, W] float in [0, 1]
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "y": self.y[idx],
            "mask": self.mask[idx],
            "weight": self.weight[idx],
            "index": torch.tensor(idx, dtype=torch.long),
        }


def _center_crop_resize(image: torch.Tensor, size: int) -> torch.Tensor:
    _, height, width = image.shape
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    image = image[:, top : top + side, left : left + side]
    if side != size:
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(0)
    return image


def _rotate(image: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate with an affine grid (keeps everything in torch, no PIL)."""
    theta_rad = np.deg2rad(degrees)
    cos_t, sin_t = float(np.cos(theta_rad)), float(np.sin(theta_rad))
    theta = torch.tensor([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0]], dtype=image.dtype).unsqueeze(0)
    grid = torch.nn.functional.affine_grid(theta, [1, *image.shape], align_corners=False)
    out = torch.nn.functional.grid_sample(
        image.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return out.squeeze(0)


def make_loaders(
    manifest: pd.DataFrame,
    images: Dict[str, np.ndarray],
    tokenize: Optional[TokenizeFn],
    cfg,
    splits: Sequence[str] = ("train", "val", "test"),
    batch_size: Optional[int] = None,
    shuffle_train: bool = True,
) -> Dict[str, DataLoader]:
    batch_size = int(batch_size or cfg.train.batch_size)
    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        rows = manifest[manifest["split"] == split]
        if len(rows) == 0:
            continue
        is_train = split == "train"
        dataset = CXRDataset(
            manifest=rows,
            images=images,
            tokenize=tokenize,
            image_size=int(cfg.data.image_size),
            train=is_train,
            modality_dropout=float(cfg.text.modality_dropout) if is_train else 0.0,
            null_text=cfg.text.empty_placeholder,
            max_length=int(cfg.text.max_length),
            synthetic_weight=float(cfg.train.synthetic_weight),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train and shuffle_train,
            num_workers=int(cfg.data.num_workers),
            pin_memory=torch.cuda.is_available(),
            drop_last=is_train and len(dataset) > batch_size,
            persistent_workers=int(cfg.data.num_workers) > 0,
        )
    return loaders


def ordered_loader(
    manifest: pd.DataFrame,
    images: Dict[str, np.ndarray],
    tokenize: Optional[TokenizeFn],
    cfg,
    batch_size: Optional[int] = None,
) -> DataLoader:
    """Deterministic, un-augmented pass over every row - used for feature
    extraction, where row order must line up with the hypergraph node order."""
    dataset = CXRDataset(
        manifest=manifest,
        images=images,
        tokenize=tokenize,
        image_size=int(cfg.data.image_size),
        train=False,
        modality_dropout=0.0,
        null_text=cfg.text.empty_placeholder,
        max_length=int(cfg.text.max_length),
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size or cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
