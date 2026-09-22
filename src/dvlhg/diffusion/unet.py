"""Label-conditional UNet for the chest-film diffusion model.

Deliberately small (~30M parameters at base 64 / 128px): on a Colab T4 this
trains to usable sample quality in well under an hour, where a full 256px
Stable-Diffusion-scale model would not finish at all. The four findings are fed
in as a multi-hot vector added to the timestep embedding, with a learned null
embedding so classifier-free guidance works.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def normalise(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, max(channels // 4, 1)), num_channels=channels)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = normalise(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # FiLM-style conditioning: the embedding produces a scale and a shift.
        self.emb = nn.Linear(emb_dim, out_channels * 2)
        self.norm2 = normalise(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb)).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = normalise(channels)
        self.heads = max(1, min(heads, channels // 32) or 1)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = self.norm(x).view(b, c, h * w)
        q, k, v = self.qkv(flat).chunk(3, dim=1)
        head_dim = c // self.heads
        q = q.view(b, self.heads, head_dim, h * w).transpose(2, 3)
        k = k.view(b, self.heads, head_dim, h * w).transpose(2, 3)
        v = v.view(b, self.heads, head_dim, h * w).transpose(2, 3)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(2, 3).reshape(b, c, h * w)
        return x + self.proj(attended).view(b, c, h, w)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16,),
        num_labels: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_labels = num_labels
        emb_dim = base_channels * 4
        self.emb_dim = emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.base_channels = base_channels
        self.label_mlp = nn.Sequential(
            nn.Linear(num_labels, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.null_label = nn.Parameter(torch.zeros(emb_dim))

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels: List[int] = [base_channels]
        channels = base_channels
        resolution = image_size
        for level, mult in enumerate(channel_mult):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(channels, out_channels, emb_dim, dropout))
                self.down_attn.append(
                    AttentionBlock(out_channels) if resolution in attn_resolutions else nn.Identity()
                )
                channels = out_channels
                skip_channels.append(channels)
            if level != len(channel_mult) - 1:
                self.downsamples.append(Downsample(channels))
                skip_channels.append(channels)
                resolution //= 2
            else:
                self.downsamples.append(nn.Identity())

        self.mid_block1 = ResBlock(channels, channels, emb_dim, dropout)
        self.mid_attn = AttentionBlock(channels)
        self.mid_block2 = ResBlock(channels, channels, emb_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_channels = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResBlock(channels + skip_channels.pop(), out_channels, emb_dim, dropout))
                self.up_attn.append(
                    AttentionBlock(out_channels) if resolution in attn_resolutions else nn.Identity()
                )
                channels = out_channels
            if level != 0:
                self.upsamples.append(Upsample(channels))
                resolution *= 2
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = normalise(channels)
        self.out_conv = nn.Conv2d(channels, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        self.num_res_blocks = num_res_blocks
        self.channel_mult = list(channel_mult)

    def conditioning(
        self, timesteps: torch.Tensor, labels: Optional[torch.Tensor], drop_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(timesteps, self.base_channels))
        if labels is None:
            return emb + self.null_label.view(1, -1)
        label_emb = self.label_mlp(labels.float())
        if drop_mask is not None:
            keep = (~drop_mask).float().unsqueeze(1)
            label_emb = keep * label_emb + (1 - keep) * self.null_label.view(1, -1)
        return emb + label_emb

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.conditioning(timesteps, labels, drop_mask)

        h = self.in_conv(x)
        skips = [h]
        block_index = 0
        for level in range(len(self.channel_mult)):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[block_index](h, emb)
                h = self.down_attn[block_index](h)
                skips.append(h)
                block_index += 1
            down = self.downsamples[level]
            if not isinstance(down, nn.Identity):
                h = down(h)
                skips.append(h)

        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        block_index = 0
        for position, _ in enumerate(self.channel_mult[::-1]):
            for _ in range(self.num_res_blocks + 1):
                h = torch.cat([h, skips.pop()], dim=1)
                h = self.up_blocks[block_index](h, emb)
                h = self.up_attn[block_index](h)
                block_index += 1
            up = self.upsamples[position]
            if not isinstance(up, nn.Identity):
                h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))
