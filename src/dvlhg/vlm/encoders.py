"""Vision-language backbone.

Three interchangeable backends behind one interface:

  biomedclip  BiomedCLIP (PubMedBERT + ViT-B/16) via open_clip. The default:
              it is pretrained on 15M biomedical image-text pairs, so its image
              and text spaces are already aligned for radiology.
  hf          timm ViT + any HuggingFace text model, assembled here. A fallback
              that does not depend on open_clip internals.
  dummy       tiny randomly initialised towers with a hashing tokenizer. No
              downloads, no network — used by the test suite and the synthetic
              smoke run.

Every backend returns the same five tensors, so nothing downstream has to care
which one is loaded:

    img_tokens [B, Ni, Dv]   patch tokens (token 0 is the CLS token)
    img_pooled [B, P]        projected image embedding
    txt_tokens [B, Nt, Dt]   contextual token states
    txt_pooled [B, P]        projected text embedding
    txt_mask   [B, Nt]       1 for real tokens, 0 for padding
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..utils import get_logger

LOG = get_logger()

# CLIP/BiomedCLIP image normalisation.
OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class VLMOutput:
    img_tokens: torch.Tensor
    img_pooled: torch.Tensor
    txt_tokens: torch.Tensor
    txt_pooled: torch.Tensor
    txt_mask: torch.Tensor


class VLMEncoder(nn.Module):
    def __init__(
        self,
        vision: nn.Module,
        text: nn.Module,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        backend: str,
        image_mean: Sequence[float] = OPENAI_MEAN,
        image_std: Sequence[float] = OPENAI_STD,
    ):
        super().__init__()
        self.vision = vision
        self.text = text
        self.vision_dim = int(vision_dim)
        self.text_dim = int(text_dim)
        self.embed_dim = int(embed_dim)
        self.backend = backend
        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(image_std).view(1, 3, 1, 1), persistent=False)

    # -- input prep --------------------------------------------------------- #
    def prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        """[B,1,H,W] or [B,3,H,W] in [0,1] -> normalised 3-channel tensor."""
        if images.ndim != 4:
            raise ValueError(f"expected [B,C,H,W] images, got {tuple(images.shape)}")
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)
        elif images.shape[1] != 3:
            raise ValueError(f"expected 1 or 3 channels, got {images.shape[1]}")
        return (images - self.image_mean) / self.image_std

    # -- forward ------------------------------------------------------------ #
    def encode_image(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.vision(self.prepare_images(images))

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.text(input_ids, attention_mask)

    def forward(
        self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> VLMOutput:
        img_tokens, img_pooled = self.encode_image(images)
        txt_tokens, txt_pooled = self.encode_text(input_ids, attention_mask)
        return VLMOutput(
            img_tokens=img_tokens,
            img_pooled=img_pooled,
            txt_tokens=txt_tokens,
            txt_pooled=txt_pooled,
            txt_mask=attention_mask.float(),
        )

    # -- partial fine-tuning ------------------------------------------------ #
    def set_trainable(
        self, vision_blocks: int = 4, text_blocks: int = 2, train_norms: bool = True
    ) -> Dict[str, int]:
        """Freeze everything, then re-enable the last N transformer blocks.

        Full fine-tuning of both towers overfits a 12k-study subset and will not
        fit a T4 at a useful batch size; the last few blocks carry almost all of
        the adaptation anyway.
        """
        for param in self.parameters():
            param.requires_grad = False

        opened = {"vision": 0, "text": 0, "norm": 0}
        for module, count, key in ((self.vision, vision_blocks, "vision"), (self.text, text_blocks, "text")):
            blocks = _find_blocks(module)
            if blocks is None:
                if count > 0:
                    LOG.warning("could not locate transformer blocks in the %s tower", key)
                continue
            if count <= 0:
                continue
            for block in blocks[-count:]:
                for param in block.parameters():
                    param.requires_grad = True
                    opened[key] += param.numel()

        if train_norms:
            for name, module in self.named_modules():
                if isinstance(module, nn.LayerNorm) and ("norm" in name.split(".")[-1] or name.endswith("norm")):
                    for param in module.parameters():
                        param.requires_grad = True
                        opened["norm"] += param.numel()

        # The projection heads are small and always worth training.
        for holder in (self.vision, self.text):
            proj = getattr(holder, "proj", None)
            if isinstance(proj, nn.Module):
                for param in proj.parameters():
                    param.requires_grad = True

        LOG.info(
            "backbone unfrozen: vision=%s params, text=%s params, norms=%s params",
            f"{opened['vision']:,}", f"{opened['text']:,}", f"{opened['norm']:,}",
        )
        return opened

    def describe(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "vision_dim": self.vision_dim,
            "text_dim": self.text_dim,
            "embed_dim": self.embed_dim,
            "vision_blocks": len(_find_blocks(self.vision) or []),
            "text_blocks": len(_find_blocks(self.text) or []),
        }


def _find_blocks(module: nn.Module) -> Optional[nn.ModuleList]:
    """Locate the transformer block list inside a tower, whatever it is called."""
    for name in ("blocks", "layer", "layers", "encoder_layers", "h", "resblocks"):
        found = _deep_get(module, name)
        if isinstance(found, (nn.ModuleList, nn.Sequential)) and len(found) > 0:
            return found  # type: ignore[return-value]
    return None


def _deep_get(module: nn.Module, attr: str, depth: int = 4):
    if hasattr(module, attr):
        return getattr(module, attr)
    if depth <= 0:
        return None
    for child in module.children():
        found = _deep_get(child, attr, depth - 1)
        if found is not None:
            return found
    return None


# --------------------------------------------------------------------------- #
# towers
# --------------------------------------------------------------------------- #
class TimmVisionTower(nn.Module):
    """timm ViT -> (patch tokens, projected embedding)."""

    def __init__(self, trunk: nn.Module, proj: Optional[nn.Module], width: int):
        super().__init__()
        self.trunk = trunk
        self.proj = proj if proj is not None else nn.Identity()
        self.width = width

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.trunk.forward_features(images)          # [B, N, D]
        if tokens.ndim == 4:  # convnet-style [B, C, H, W]
            tokens = tokens.flatten(2).transpose(1, 2)
        if hasattr(self.trunk, "forward_head"):
            pooled = self.trunk.forward_head(tokens, pre_logits=True)
        else:  # pragma: no cover - non-timm trunk
            pooled = tokens[:, 0]
        if pooled.ndim == 3:
            pooled = pooled.mean(dim=1)
        return tokens, self.proj(pooled)


class HFTextTower(nn.Module):
    """HuggingFace encoder -> (token states, projected CLS embedding)."""

    def __init__(self, transformer: nn.Module, proj: Optional[nn.Module], width: int):
        super().__init__()
        self.transformer = transformer
        self.proj = proj if proj is not None else nn.Identity()
        self.width = width

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        pooled = tokens[:, 0]  # BiomedCLIP uses the CLS position
        return tokens, self.proj(pooled)


class DummyVisionTower(nn.Module):
    """Small randomly initialised ViT. Offline, seconds to train, enough signal
    on the synthetic films to make the smoke test meaningful."""

    def __init__(self, image_size: int = 224, patch: int = 16, width: int = 192, depth: int = 4, embed_dim: int = 128):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, width, kernel_size=patch, stride=patch)
        grid = image_size // patch
        self.num_patches = grid * grid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=width * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.ModuleList([layer] + [
            nn.TransformerEncoderLayer(
                d_model=width, nhead=4, dim_feedforward=width * 4,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(depth - 1)
        ])
        self.norm = nn.LayerNorm(width)
        self.proj = nn.Linear(width, embed_dim)
        self.width = width

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        if x.shape[1] == self.pos_embed.shape[1]:
            x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x, self.proj(x[:, 0])


class DummyTextTower(nn.Module):
    def __init__(self, vocab: int = 30522, width: int = 192, depth: int = 2, embed_dim: int = 128, max_len: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, max_len, width) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=width, nhead=4, dim_feedforward=width * 4,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width)
        self.proj = nn.Linear(width, embed_dim)
        self.width = width

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(input_ids) + self.pos[:, : input_ids.shape[1]]
        pad = attention_mask == 0
        for block in self.blocks:
            x = block(x, src_key_padding_mask=pad)
        x = self.norm(x)
        return x, self.proj(x[:, 0])


# --------------------------------------------------------------------------- #
# tokenizers
# --------------------------------------------------------------------------- #
def _wrap_hf_tokenizer(tokenizer, max_length: int) -> Callable[[Sequence[str]], Dict[str, torch.Tensor]]:
    def tokenize(texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        encoded = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].long(),
            "attention_mask": encoded["attention_mask"].long(),
        }

    return tokenize


def _hashing_tokenizer(max_length: int, vocab: int = 30522) -> Callable[[Sequence[str]], Dict[str, torch.Tensor]]:
    """Deterministic word-hash tokenizer. id 0 is reserved for padding."""

    def tokenize(texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        ids = torch.zeros((len(texts), max_length), dtype=torch.long)
        mask = torch.zeros((len(texts), max_length), dtype=torch.long)
        for row, text in enumerate(texts):
            words = ["[cls]"] + str(text).lower().split()[: max_length - 1]
            for col, word in enumerate(words):
                digest = hashlib.md5(word.encode("utf-8")).digest()
                ids[row, col] = 1 + int.from_bytes(digest[:4], "little") % (vocab - 1)
                mask[row, col] = 1
        return {"input_ids": ids, "attention_mask": mask}

    return tokenize


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def build_vlm(cfg) -> Tuple[VLMEncoder, Callable[[Sequence[str]], Dict[str, torch.Tensor]]]:
    """Build the encoder named by cfg.vlm.backbone, with a clear fallback chain."""
    backend = str(cfg.vlm.backbone).lower()
    max_length = int(cfg.text.max_length)

    if backend == "biomedclip":
        try:
            return _build_biomedclip(cfg, max_length)
        except Exception as exc:  # noqa: BLE001
            LOG.error("BiomedCLIP could not be loaded (%s)", exc)
            LOG.error(
                "falling back to the 'hf' backend. To use BiomedCLIP, make sure "
                "`pip install open_clip_torch>=2.24` succeeded and the machine can "
                "reach huggingface.co."
            )
            backend = "hf"

    if backend == "hf":
        try:
            return _build_hf(cfg, max_length)
        except Exception as exc:  # noqa: BLE001
            LOG.error("HF backend could not be loaded (%s); falling back to 'dummy'", exc)
            backend = "dummy"

    if backend == "dummy":
        return _build_dummy(cfg, max_length)

    raise ValueError(f"unknown vlm.backbone '{cfg.vlm.backbone}'")


def _build_biomedclip(cfg, max_length: int) -> Tuple[VLMEncoder, Callable]:
    import open_clip

    hub = str(cfg.vlm.hf_hub)
    LOG.info("loading %s", hub)
    model = open_clip.create_model(hub)
    tokenizer = open_clip.get_tokenizer(hub)

    visual = model.visual
    trunk = getattr(visual, "trunk", None)
    if trunk is None:
        raise RuntimeError("expected open_clip TimmModel visual tower with a .trunk")
    vision_proj = getattr(visual, "head", None)
    vision_width = int(getattr(trunk, "num_features", 768))

    text_module = model.text
    transformer = getattr(text_module, "transformer", None)
    if transformer is None:
        raise RuntimeError("expected an open_clip HFTextEncoder with a .transformer")
    text_proj = getattr(text_module, "proj", None)
    text_width = int(getattr(transformer.config, "hidden_size", 768))
    embed_dim = int(cfg.vlm.embed_dim)

    vision_tower = TimmVisionTower(trunk, vision_proj, vision_width)
    text_tower = HFTextTower(transformer, text_proj, text_width)

    # open_clip's HFTokenizer pads to context_length and returns ids only.
    hf_tok = getattr(tokenizer, "tokenizer", None)
    pad_id = int(getattr(hf_tok, "pad_token_id", 0) or 0)
    context_length = min(max_length, int(getattr(tokenizer, "context_length", max_length) or max_length))

    def tokenize(texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        if hf_tok is not None:
            encoded = hf_tok(
                list(texts), padding="max_length", truncation=True,
                max_length=context_length, return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].long(),
                "attention_mask": encoded["attention_mask"].long(),
            }
        ids = tokenizer(list(texts), context_length=context_length)
        if isinstance(ids, (tuple, list)):
            ids = ids[0]
        ids = ids.long()
        return {"input_ids": ids, "attention_mask": (ids != pad_id).long()}

    encoder = VLMEncoder(
        vision=vision_tower, text=text_tower,
        vision_dim=vision_width, text_dim=text_width,
        embed_dim=embed_dim, backend="biomedclip",
    )
    LOG.info("BiomedCLIP ready: %s", encoder.describe())
    return encoder, tokenize


def _build_hf(cfg, max_length: int) -> Tuple[VLMEncoder, Callable]:
    import timm
    from transformers import AutoModel, AutoTokenizer

    vision_name = str(cfg.vlm.get("timm_model", "vit_base_patch16_224"))
    text_name = str(cfg.vlm.get("hf_text_model", "emilyalsentzer/Bio_ClinicalBERT"))
    embed_dim = int(cfg.vlm.embed_dim)

    trunk = timm.create_model(vision_name, pretrained=True, num_classes=0)
    vision_width = int(trunk.num_features)
    vision_tower = TimmVisionTower(trunk, nn.Linear(vision_width, embed_dim), vision_width)

    transformer = AutoModel.from_pretrained(text_name)
    text_width = int(transformer.config.hidden_size)
    text_tower = HFTextTower(transformer, nn.Linear(text_width, embed_dim), text_width)

    tokenizer = AutoTokenizer.from_pretrained(text_name)
    encoder = VLMEncoder(
        vision=vision_tower, text=text_tower,
        vision_dim=vision_width, text_dim=text_width,
        embed_dim=embed_dim, backend="hf",
    )
    LOG.info("HF backbone ready: %s + %s -> %s", vision_name, text_name, encoder.describe())
    return encoder, _wrap_hf_tokenizer(tokenizer, max_length)


def _build_dummy(cfg, max_length: int) -> Tuple[VLMEncoder, Callable]:
    embed_dim = int(cfg.vlm.get("dummy_embed_dim", 128))
    width = int(cfg.vlm.get("dummy_width", 192))
    vision_tower = DummyVisionTower(
        image_size=int(cfg.data.image_size), width=width, embed_dim=embed_dim
    )
    text_tower = DummyTextTower(width=width, embed_dim=embed_dim, max_len=max_length)
    encoder = VLMEncoder(
        vision=vision_tower, text=text_tower,
        vision_dim=width, text_dim=width,
        embed_dim=embed_dim, backend="dummy",
        image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
    )
    LOG.warning("using the DUMMY backbone - random weights, for tests only")
    return encoder, _hashing_tokenizer(max_length)
