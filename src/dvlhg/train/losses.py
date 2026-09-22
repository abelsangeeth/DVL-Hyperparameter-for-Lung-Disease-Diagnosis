"""Multi-label losses with per-cell masking.

Masking matters here: with `uncertain_policy: ignore`, a cell the radiologist
was unsure about contributes no gradient at all, rather than being forced to 0
or 1. Every loss in this file respects that mask, and every one also accepts a
per-sample weight so diffusion-generated rows can count for less than real ones.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _reduce(loss: torch.Tensor, mask: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
    """Weighted mean over the cells that count.

        sum_ij  w_i * mask_ij * loss_ij  /  sum_ij  w_i * mask_ij

    The weight is folded into the mask once and once only. Applying it to the
    numerator as well squares it: uniform weights would then rescale the loss,
    and a synthetic row at weight 0.5 would really count 0.25.
    """
    if weight is not None:
        mask = mask * weight.view(-1, 1)
    denominator = mask.sum().clamp_min(1e-6)
    return (loss * mask).sum() / denominator


class MaskedBCE(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer(
            "pos_weight", pos_weight if pos_weight is not None else torch.empty(0), persistent=False
        )
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        pos_weight = self.pos_weight if self.pos_weight.numel() else None
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=pos_weight.to(logits.device) if pos_weight is not None else None,
        )
        return _reduce(loss, mask, weight)


class AsymmetricLoss(nn.Module):
    """Ridnik et al., "Asymmetric Loss For Multi-Label Classification".

    Down-weights easy negatives (gamma_neg) and hard-clips very low-probability
    negatives away entirely (clip). On chest X-ray label sets, where negatives
    outnumber positives 5-20x, this consistently beats weighted BCE.
    """

    def __init__(self, gamma_neg: float = 3.0, gamma_pos: float = 0.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_neg = probs
        if self.clip > 0:
            probs_neg = (probs - self.clip).clamp(min=0)

        loss_pos = targets * torch.log(probs.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=self.eps))

        # The focusing term is detached: gradients flow through the log only,
        # which is what the reference implementation does and what keeps
        # training stable at gamma_neg > 2.
        with torch.no_grad():
            pt = probs * targets + probs_neg * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            focus = torch.pow(1 - pt, gamma)

        loss = -(loss_pos + loss_neg) * focus
        return _reduce(loss, mask, weight)


def build_loss(cfg, pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    kind = str(cfg.train.loss).lower()
    if kind == "asl":
        return AsymmetricLoss(
            gamma_neg=float(cfg.train.asl_gamma_neg),
            gamma_pos=float(cfg.train.asl_gamma_pos),
            clip=float(cfg.train.asl_clip),
        )
    if kind == "bce":
        return MaskedBCE(pos_weight=pos_weight, label_smoothing=float(cfg.train.label_smoothing))
    raise ValueError(f"unknown train.loss '{cfg.train.loss}' (bce | asl)")
