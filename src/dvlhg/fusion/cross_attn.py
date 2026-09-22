"""Bidirectional cross-attention fusion of image patches and report tokens.

Each block lets every patch attend over the report and every report token attend
over the film, so "decreased breath sounds on the right" can bind to the right
costophrenic angle rather than to a global average. The two streams are then
attention-pooled and combined with a gate, which is what keeps the model usable
when one modality is missing.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Pool a token sequence with a single learned query."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        query = self.query.expand(tokens.shape[0], -1, -1)
        key_padding = None
        if mask is not None:
            key_padding = mask < 0.5  # True marks positions to ignore
        pooled, _ = self.attn(query, tokens, tokens, key_padding_mask=key_padding, need_weights=False)
        return self.norm(pooled.squeeze(1))


class CrossBlock(nn.Module):
    """One pre-LN cross-attention + feed-forward step for a single stream."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        key_padding = None
        if context_mask is not None:
            key_padding = context_mask < 0.5
        attended, weights = self.attn(
            self.norm_q(query),
            self.norm_kv(context),
            self.norm_kv(context),
            key_padding_mask=key_padding,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        query = query + self.drop(attended)
        query = query + self.drop(self.ff(self.norm_ff(query)))
        return query, weights


class CrossModalFusion(nn.Module):
    """image tokens + report tokens -> one node feature per study."""

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        dim: int = 384,
        layers: int = 2,
        heads: int = 6,
        dropout: float = 0.1,
        token_level: bool = True,
        max_image_tokens: int = 197,
        pool: str = "attention",
    ):
        super().__init__()
        self.dim = dim
        self.token_level = token_level
        self.max_image_tokens = max_image_tokens
        self.pool_mode = pool

        self.img_proj = nn.Linear(vision_dim, dim)
        self.txt_proj = nn.Linear(text_dim, dim)
        self.img_norm = nn.LayerNorm(dim)
        self.txt_norm = nn.LayerNorm(dim)
        self.modality_embed = nn.Parameter(torch.randn(2, dim) * 0.02)

        if token_level:
            self.img_blocks = nn.ModuleList([CrossBlock(dim, heads, dropout) for _ in range(layers)])
            self.txt_blocks = nn.ModuleList([CrossBlock(dim, heads, dropout) for _ in range(layers)])
            self.img_pool = AttentionPool(dim, heads=max(heads // 2, 1), dropout=dropout)
            self.txt_pool = AttentionPool(dim, heads=max(heads // 2, 1), dropout=dropout)

        # Pooled projected embeddings from the backbone go in as a shortcut, so
        # the fusion trunk never has to relearn what BiomedCLIP already aligned.
        self.emb_proj = nn.Linear(embed_dim * 2, dim)

        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.combine = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        img_tokens: torch.Tensor,
        img_pooled: torch.Tensor,
        txt_tokens: torch.Tensor,
        txt_pooled: torch.Tensor,
        txt_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        extras: dict = {}

        # A fully masked sequence makes MultiheadAttention emit NaNs, so keep
        # the first position alive no matter what upstream produced.
        txt_mask = txt_mask.clone().float()
        txt_mask[:, 0] = 1.0

        if self.token_level:
            image = self.img_norm(self.img_proj(img_tokens[:, : self.max_image_tokens]))
            text = self.txt_norm(self.txt_proj(txt_tokens))
            image = image + self.modality_embed[0].view(1, 1, -1)
            text = text + self.modality_embed[1].view(1, 1, -1)
            text_mask = txt_mask[:, : text.shape[1]]

            last_weights = None
            for img_block, txt_block in zip(self.img_blocks, self.txt_blocks):
                new_image, weights = img_block(image, text, text_mask, need_weights=return_attention)
                new_text, _ = txt_block(text, image, None)
                image, text = new_image, new_text
                last_weights = weights if weights is not None else last_weights

            h_img = self.img_pool(image)
            h_txt = self.txt_pool(text, text_mask)
            if return_attention:
                extras["text_attention"] = last_weights            # [B, Ni, Nt]
                extras["image_tokens"] = image                      # [B, Ni, d]
        else:
            h_img = self.img_norm(self.img_proj(img_tokens[:, 0]))
            masked = txt_tokens * txt_mask.unsqueeze(-1)
            mean_text = masked.sum(dim=1) / txt_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            h_txt = self.txt_norm(self.txt_proj(mean_text))

        gate = self.gate(torch.cat([h_img, h_txt], dim=-1))
        gated = gate * h_img + (1.0 - gate) * h_txt
        interaction = torch.cat([h_img, h_txt, h_img * h_txt, (h_img - h_txt).abs()], dim=-1)
        node = self.combine(interaction) + gated
        node = node + self.emb_proj(torch.cat([img_pooled, txt_pooled], dim=-1))

        extras["gate"] = gate.mean(dim=-1)   # 1 = leaning on the image, 0 = on the report
        extras["h_image"] = h_img
        extras["h_text"] = h_txt
        return self.out_norm(node), extras


class ClassifierHead(nn.Module):
    """Node feature -> four logits.

    `zero_init` zeroes the last layer so the head starts by predicting exactly
    zero. That is what lets the hypergraph stage begin as an identity on top of
    the backbone's logits and learn a correction, rather than relearning the
    whole decision from scratch on a few thousand nodes.
    """

    def __init__(self, dim: int, num_labels: int = 4, dropout: float = 0.1,
                 hidden: int = 0, zero_init: bool = False):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, num_labels),
            )
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_labels))
        if zero_init:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, node: torch.Tensor) -> torch.Tensor:
        return self.net(node)
