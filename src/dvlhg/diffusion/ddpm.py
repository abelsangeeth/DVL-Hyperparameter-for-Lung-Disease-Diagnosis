"""DDPM training objective + DDIM sampling, with classifier-free guidance.

Two uses downstream, both configured under `diffusion.use`:

  synthesis  sample new films from a label vector and add them to the training
             split. This is what helps the rare combinations (edema without
             effusion, three findings at once) that a 12k subset barely covers.
  refine     SDEdit: noise a real film to a fraction of the schedule and denoise
             it back. The anatomy survives, the texture and acquisition
             characteristics change — a learned augmentation that respects the
             image prior instead of jittering pixels blindly.

Images live in [-1, 1] throughout this module.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(kind: str, timesteps: int, s: float = 0.008) -> torch.Tensor:
    if kind == "linear":
        scale = 1000 / timesteps
        return torch.linspace(scale * 1e-4, scale * 0.02, timesteps, dtype=torch.float64)
    if kind == "cosine":
        steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        alphas_cumprod = torch.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return betas.clamp(1e-8, 0.999)
    raise ValueError(f"unknown schedule '{kind}' (cosine | linear)")


class GaussianDiffusion(nn.Module):
    """Schedule bookkeeping, the training loss, and the samplers."""

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        schedule: str = "cosine",
        objective: str = "eps",
    ):
        super().__init__()
        self.model = model
        self.timesteps = int(timesteps)
        self.objective = objective
        if objective not in ("eps", "v"):
            raise ValueError("objective must be 'eps' or 'v'")

        betas = make_beta_schedule(schedule, self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        register = lambda name, value: self.register_buffer(  # noqa: E731
            name, value.to(torch.float32), persistent=False
        )
        register("betas", betas)
        register("alphas_cumprod", alphas_cumprod)
        register("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        register("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

    # -- forward process ---------------------------------------------------- #
    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        noise = torch.randn_like(x_start) if noise is None else noise
        shape = (-1,) + (1,) * (x_start.ndim - 1)
        return (
            self.sqrt_alphas_cumprod[t].view(shape) * x_start
            + self.sqrt_one_minus_alphas_cumprod[t].view(shape) * noise
        )

    def target_for(self, x_start: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.objective == "eps":
            return noise
        shape = (-1,) + (1,) * (x_start.ndim - 1)
        return (
            self.sqrt_alphas_cumprod[t].view(shape) * noise
            - self.sqrt_one_minus_alphas_cumprod[t].view(shape) * x_start
        )

    def loss(
        self,
        x_start: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        cond_dropout: float = 0.1,
    ) -> torch.Tensor:
        batch = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (batch,), device=x_start.device)
        noise = torch.randn_like(x_start)
        noisy = self.q_sample(x_start, t, noise)
        drop_mask = None
        if labels is not None and cond_dropout > 0:
            drop_mask = torch.rand(batch, device=x_start.device) < cond_dropout
        prediction = self.model(noisy, t, labels, drop_mask)
        return F.mse_loss(prediction, self.target_for(x_start, noise, t))

    # -- reverse process ---------------------------------------------------- #
    def _predict(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor],
        guidance_scale: float,
    ) -> torch.Tensor:
        """Model output with classifier-free guidance folded in."""
        if labels is None or guidance_scale == 1.0:
            return self.model(x, t, labels)
        # One batched pass with the conditional half and the null half.
        doubled_x = torch.cat([x, x], dim=0)
        doubled_t = torch.cat([t, t], dim=0)
        doubled_labels = torch.cat([labels, labels], dim=0)
        drop = torch.cat(
            [torch.zeros(len(x), dtype=torch.bool, device=x.device),
             torch.ones(len(x), dtype=torch.bool, device=x.device)],
            dim=0,
        )
        both = self.model(doubled_x, doubled_t, doubled_labels, drop)
        conditional, unconditional = both.chunk(2, dim=0)
        return unconditional + guidance_scale * (conditional - unconditional)

    def _to_eps_and_x0(
        self, prediction: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = (-1,) + (1,) * (x.ndim - 1)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(shape)
        sqrt_1mac = self.sqrt_one_minus_alphas_cumprod[t].view(shape)
        if self.objective == "eps":
            eps = prediction
            x0 = (x - sqrt_1mac * eps) / sqrt_ac.clamp_min(1e-8)
        else:  # v-prediction
            x0 = sqrt_ac * x - sqrt_1mac * prediction
            eps = sqrt_ac * prediction + sqrt_1mac * x
        return eps, x0.clamp(-1.0, 1.0)

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: Sequence[int],
        labels: Optional[torch.Tensor] = None,
        steps: int = 50,
        guidance_scale: float = 2.0,
        eta: float = 0.0,
        device: Optional[torch.device] = None,
        x_start: Optional[torch.Tensor] = None,
        start_step: Optional[int] = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """Deterministic DDIM (eta=0) by default.

        `x_start` + `start_step` turn this into SDEdit: begin from a noised real
        film partway down the schedule instead of from pure noise.
        """
        device = device or next(self.model.parameters()).device
        total = self.timesteps if start_step is None else int(start_step)
        total = max(min(total, self.timesteps), 1)
        step_count = max(min(int(steps), total), 1)
        schedule = np.linspace(0, total - 1, step_count).round().astype(int)[::-1]

        if x_start is not None:
            t0 = torch.full((shape[0],), int(schedule[0]), device=device, dtype=torch.long)
            x = self.q_sample(x_start.to(device), t0)
        else:
            x = torch.randn(*shape, device=device)

        iterator = enumerate(schedule)
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(list(iterator), total=len(schedule), desc="ddim", leave=False)
            except ImportError:
                pass

        for position, step in iterator:
            t = torch.full((x.shape[0],), int(step), device=device, dtype=torch.long)
            prediction = self._predict(x, t, labels, guidance_scale)
            eps, x0 = self._to_eps_and_x0(prediction, x, t)

            previous = int(schedule[position + 1]) if position + 1 < len(schedule) else -1
            alpha_previous = (
                self.alphas_cumprod[previous] if previous >= 0
                else torch.tensor(1.0, device=device)
            )
            sigma = 0.0
            if eta > 0 and previous >= 0:
                alpha_current = self.alphas_cumprod[step]
                sigma = float(
                    eta
                    * torch.sqrt((1 - alpha_previous) / (1 - alpha_current))
                    * torch.sqrt(1 - alpha_current / alpha_previous)
                )
            direction = torch.sqrt((1 - alpha_previous - sigma**2).clamp_min(0.0)) * eps
            x = torch.sqrt(alpha_previous) * x0 + direction
            if sigma > 0:
                x = x + sigma * torch.randn_like(x)

        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def refine(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        strength: float = 0.25,
        steps: int = 20,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """SDEdit refinement of real films (augmentation / test-time smoothing)."""
        start_step = max(int(self.timesteps * float(strength)), 1)
        return self.ddim_sample(
            shape=tuple(images.shape),
            labels=labels,
            steps=steps,
            guidance_scale=guidance_scale,
            device=images.device,
            x_start=images,
            start_step=start_step,
        )


def to_model_space(images_01: torch.Tensor) -> torch.Tensor:
    """[0, 1] -> [-1, 1]"""
    return images_01 * 2.0 - 1.0


def to_image_space(images_pm1: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]"""
    return ((images_pm1 + 1.0) / 2.0).clamp(0.0, 1.0)
