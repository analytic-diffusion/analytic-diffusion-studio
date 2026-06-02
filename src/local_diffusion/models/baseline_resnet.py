from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler

from local_diffusion.models import register_model
from local_diffusion.models.base import BaseDenoiser


class ResidualLayer(nn.Module):
    def __init__(self, channels: int, padding_mode: str = "zeros") -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, padding_mode=padding_mode)
        self.time = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x + F.relu(self.conv(F.relu(x)) + self.time(t)[:, :, None, None])


class LocalResNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 128,
        num_layers: int = 8,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )
        self.in_conv = nn.Conv2d(in_channels, channels, 3, padding=1, padding_mode=padding_mode)
        self.in_time = nn.Linear(channels, channels)
        self.layers = nn.ModuleList([ResidualLayer(channels, padding_mode) for _ in range(num_layers - 1)])
        self.out_conv = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(t[:, None])
        h = F.relu(self.in_conv(x) + self.in_time(t_embed)[:, :, None, None])
        for layer in self.layers:
            h = layer(h, t_embed)
        return self.out_conv(F.relu(h))


@register_model("baseline_resnet")
class BaselineResNet(BaseDenoiser):
    def __init__(
        self,
        resolution: int,
        device: str,
        num_steps: int,
        model_path: str | None = None,
        dataset_name: str = "mnist",
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 128,
        num_layers: int = 8,
        padding_mode: str = "zeros",
        scheduler_train_steps: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            resolution=resolution,
            device=device,
            num_steps=num_steps,
            dataset_name=dataset_name,
            in_channels=in_channels,
            **kwargs,
        )

        self.model = LocalResNet(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            num_layers=num_layers,
            padding_mode=padding_mode,
        ).to(self.device)

        self.scheduler = DDIMScheduler(
            num_train_timesteps=scheduler_train_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        self.scheduler.set_timesteps(num_steps)

        if model_path is not None:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)

    @staticmethod
    def alpha_sigma(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = 0.008
        alpha_bar = torch.cos((t + s) / (1 + s) * torch.pi / 2).square()
        alpha_bar = alpha_bar / np.cos(s / (1 + s) * np.pi / 2) ** 2
        alpha_bar = alpha_bar.clamp(0, 1)
        return alpha_bar.sqrt(), (1 - alpha_bar).sqrt()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)

    @torch.no_grad()
    def denoise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        generator: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        b = latents.shape[0]

        if isinstance(timestep, (int, float)):
            ts = torch.full((b,), int(timestep), device=self.device, dtype=torch.long)
        elif timestep.ndim == 0:
            ts = timestep.expand(b).long().to(self.device)
        else:
            ts = timestep.long().to(self.device)

        t_cont = ts.float() / (self.scheduler.config.num_train_timesteps - 1)
        v = self.model(latents, t_cont)

        alpha_prod = self.scheduler.alphas_cumprod[ts.cpu().long()].to(self.device)
        alpha = alpha_prod.sqrt()[:, None, None, None]
        sigma = (1 - alpha_prod).sqrt()[:, None, None, None]
        return alpha * latents - sigma * v

    def train(self, dataset: Any) -> None:
        pass
