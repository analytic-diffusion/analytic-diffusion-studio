"""Smoothed variant of the optimal denoiser baseline."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from local_diffusion.data import DatasetBundle
from local_diffusion.models import register_model
from local_diffusion.models.optimal import OptimalDenoiser


@register_model("scfdm")
class SmoothedCFDM(OptimalDenoiser):
    """Optimal denoiser averaged over Gaussian-perturbed latent copies."""

    def __init__(
        self,
        dataset: DatasetBundle,
        device: str,
        num_steps: int,
        *,
        params: Optional[Dict[str, object]] = None,
        **kwargs: Any,
    ) -> None:
        params = params or {}
        super().__init__(
            dataset=dataset,
            device=device,
            num_steps=num_steps,
            params=params,
            **kwargs,
        )

        self.num_noise = int(params.get("num_noise", 1))
        self.smoothing_std = float(params.get("smoothing_std", 0.0))

        if self.num_noise <= 0:
            raise ValueError("num_noise must be a positive integer")
        if self.smoothing_std < 0:
            raise ValueError("smoothing_std must be non-negative")

    @torch.no_grad()
    def denoise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch_shape = (self.num_noise, *latents.shape)
        noise = torch.randn(
            batch_shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        smoothed_latents = latents.unsqueeze(0) + self.smoothing_std * noise
        flat_latents = smoothed_latents.reshape(-1, *latents.shape[1:])

        pred_x0 = super().denoise(
            flat_latents,
            timestep,
            generator=generator,
            **kwargs,
        )
        return pred_x0.reshape(self.num_noise, *latents.shape).mean(dim=0)
