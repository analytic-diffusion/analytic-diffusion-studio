from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from diffusers import DDIMScheduler
from tqdm import tqdm


@dataclass
class SamplingOutput:
    images: torch.Tensor
    timesteps: Optional[List[int]]
    trajectory_xt: Optional[List[torch.Tensor]]
    trajectory_x0: Optional[List[torch.Tensor]]


class BaseDenoiser(torch.nn.Module):
    """Base diffusion interface shared by analytic and learned models."""

    prediction_type: str = "epsilon"

    def __init__(
        self,
        resolution: int,
        device: str,
        num_steps: int,
        *args,
        beta_1: float = 0.0001,
        beta_T: float = 0.02,
        dataset_name: str = "cifar10",
        scheduler_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.n_channels = kwargs.get("in_channels", 3)
        self.img_resolution = resolution
        self.resolution = resolution
        self.dataset_name = dataset_name
        self.num_steps = num_steps

        scheduler_kwargs = scheduler_kwargs or {}
        self.scheduler = DDIMScheduler(
            beta_start=beta_1,
            beta_end=beta_T,
            beta_schedule="linear",
            prediction_type=self.prediction_type,
            **scheduler_kwargs,
        )
        self.scheduler.set_timesteps(num_steps)

    def denoise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Predict the denoised sample ``x_0`` for a given latent tensor.

        Sub-classes **must** override this method to implement their analytic denoiser.
        The return value should be the predicted clean sample together with any
        auxiliary information that should be tracked (e.g. nearest-neighbour indices).
        """
        raise NotImplementedError

    @torch.no_grad()
    def denoise_sigma(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Denoise an EDM-convention noisy input ``x = x0 + sigma * eps`` at level ``sigma``.

        Returns the clean-image estimate ``x0_hat = D(x; sigma)``. This is the interface the
        EDM Heun sampler uses. The default implementation bridges to the DDPM-timestep
        :meth:`denoise` via the VP correspondence: with ``alpha_bar = 1/(1 + sigma**2)`` the
        DDPM-scaled latent is ``x_t = sqrt(alpha_bar) * x`` at the nearest scheduler timestep.
        Sub-classes with a native sigma denoiser (e.g. ``edm_unet``) should override this.
        """
        sigma_val = float(sigma)
        if sigma_val <= 0.0:
            return self.denoise(x, torch.tensor(0, device=x.device))

        alpha_bar = 1.0 / (1.0 + sigma_val * sigma_val)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(x.device)
        t = int(torch.argmin((alphas_cumprod - alpha_bar).abs()).item())
        x_t = x * (alpha_bar ** 0.5)
        return self.denoise(x_t, torch.tensor(t, device=x.device))

    def build_sample_output(
        self,
        images: torch.Tensor,
        trajectory_xt: Optional[List[torch.Tensor]],
        trajectory_x0: Optional[List[torch.Tensor]],
        timesteps: Optional[List[int]],
    ) -> SamplingOutput:
        return SamplingOutput(
            images=images,
            trajectory_xt=trajectory_xt,
            trajectory_x0=trajectory_x0,
            timesteps=timesteps,
        )

    def train(self, dataset):
        raise NotImplementedError

    def set_timesteps(self, num_steps: int) -> None:
        self.scheduler.set_timesteps(num_steps)
        self.num_steps = num_steps

    def prepare_latents(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        shape = (batch_size, self.n_channels, self.resolution, self.resolution)
        latents = torch.randn(shape, generator=generator, device=self.device)
        return latents * self.scheduler.init_noise_sigma

    def compute_noise_from_x0(
        self,
        x_t: torch.Tensor,
        pred_x0: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        t = int(timestep.item() if isinstance(timestep, torch.Tensor) else timestep)
        alpha_prod = self.scheduler.alphas_cumprod[t].to(x_t.device)
        beta_prod = 1 - alpha_prod
        sqrt_alpha = torch.sqrt(alpha_prod)
        sqrt_beta = torch.sqrt(beta_prod + 1e-8)
        return (x_t - sqrt_alpha * pred_x0) / sqrt_beta

    @torch.no_grad()
    def _image_preprocess(self, img: torch.Tensor) -> torch.Tensor:
        imgs = torch.nn.functional.interpolate(
            img[:, : self.n_channels, ...].to(self.device),
            size=(self.img_resolution, self.img_resolution),
            mode="bilinear",
            align_corners=False,
        )
        img_rescaled = (imgs - 0.5) * 2
        return img_rescaled

    @torch.no_grad()
    def _image_postprocess(self, img: torch.Tensor) -> torch.Tensor:
        img_rescaled = (img + 1) / 2
        return img_rescaled.clamp(0, 1)

    @torch.no_grad()
    def sample(
        self,
        *,
        num_samples: int,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        return_intermediates: bool = False,
        method: str = "ddim",
        sampler_kwargs: Optional[Dict[str, Any]] = None,
    ) -> SamplingOutput:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        batches: List[SamplingOutput] = []
        total_generated = 0
        while total_generated < num_samples:
            current_batch = min(batch_size, num_samples - total_generated)
            batch_result = self._sample_batch(
                batch_size=current_batch,
                generator=generator,
                return_intermediates=return_intermediates,
                method=method,
                sampler_kwargs=sampler_kwargs,
            )
            batches.append(batch_result)
            total_generated += current_batch

        images = torch.cat([b.images for b in batches], dim=0)

        trajectory_xt: Optional[List[torch.Tensor]] = None
        trajectory_x0: Optional[List[torch.Tensor]] = None
        timesteps: Optional[List[int]] = None

        if return_intermediates:
            for batch in batches:
                if batch.trajectory_xt is not None:
                    if trajectory_xt is None:
                        trajectory_xt = [tensor.clone() for tensor in batch.trajectory_xt]
                    else:
                        for idx, tensor in enumerate(batch.trajectory_xt):
                            trajectory_xt[idx] = torch.cat([trajectory_xt[idx], tensor], dim=0)

                if batch.trajectory_x0 is not None:
                    if trajectory_x0 is None:
                        trajectory_x0 = [tensor.clone() for tensor in batch.trajectory_x0]
                    else:
                        for idx, tensor in enumerate(batch.trajectory_x0):
                            trajectory_x0[idx] = torch.cat([trajectory_x0[idx], tensor], dim=0)

                if batch.timesteps is not None and timesteps is None:
                    timesteps = list(batch.timesteps)

        return self.build_sample_output(
            images=images,
            trajectory_xt=trajectory_xt,
            trajectory_x0=trajectory_x0,
            timesteps=timesteps,
        )

    # Default EDM Heun sampler settings (Karras et al. 2022). Sub-classes may narrow the
    # sigma range by setting ``sampler_sigma_min`` / ``sampler_sigma_max`` (edm_unet does).
    _HEUN_DEFAULTS: Dict[str, float] = {
        "sigma_min": 0.002,
        "sigma_max": 80.0,
        "rho": 7.0,
        "s_churn": 0.0,
        "s_min": 0.0,
        "s_max": float("inf"),
        "s_noise": 1.0,
    }

    def _sample_batch(
        self,
        *,
        batch_size: int,
        generator: Optional[torch.Generator],
        return_intermediates: bool,
        method: str = "ddim",
        sampler_kwargs: Optional[Dict[str, Any]] = None,
    ) -> SamplingOutput:
        if method == "heun":
            params = dict(self._HEUN_DEFAULTS)
            if sampler_kwargs:
                params.update({k: v for k, v in sampler_kwargs.items() if k in params})
            return self._sample_batch_heun(
                batch_size=batch_size,
                generator=generator,
                return_intermediates=return_intermediates,
                **params,
            )
        if method not in ("ddim", None):
            raise ValueError(f"Unknown sampling method '{method}' (expected 'ddim' or 'heun')")
        return self._sample_batch_ddim(
            batch_size=batch_size,
            generator=generator,
            return_intermediates=return_intermediates,
        )

    def _sample_batch_ddim(
        self,
        *,
        batch_size: int,
        generator: Optional[torch.Generator],
        return_intermediates: bool,
    ) -> SamplingOutput:
        latents = self.prepare_latents(batch_size, generator=generator)

        trajectory_xt: Optional[List[torch.Tensor]] = [] if return_intermediates else None
        trajectory_x0: Optional[List[torch.Tensor]] = [] if return_intermediates else None
        timesteps_list: Optional[List[int]] = [] if return_intermediates else None
        last_pred_x0: Optional[torch.Tensor] = None

        timesteps_iter = tqdm(
            enumerate(self.scheduler.timesteps),
            total=len(self.scheduler.timesteps),
            desc="Sampling",
            unit="step",
        )
        for step_idx, timestep in timesteps_iter:
            pred_x0 = self.denoise(
                latents,
                timestep,
                generator=generator,
            )

            if not isinstance(pred_x0, torch.Tensor):
                raise TypeError("denoise must return a torch.Tensor as the first value")

            predicted_noise = self.compute_noise_from_x0(latents, pred_x0, timestep)
            step_output = self.scheduler.step(
                model_output=predicted_noise,
                timestep=timestep,
                sample=latents,
            )

            if return_intermediates:
                if trajectory_xt is not None:
                    trajectory_xt.append(latents.detach().cpu())
                if trajectory_x0 is not None:
                    trajectory_x0.append(pred_x0.detach().cpu())
                if timesteps_list is not None:
                    timestep_value = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
                    timesteps_list.append(timestep_value)

            last_pred_x0 = pred_x0
            latents = step_output.prev_sample

        if last_pred_x0 is None:
            raise RuntimeError("Sampling loop did not execute any timesteps.")

        return SamplingOutput(
            images=last_pred_x0.detach().cpu(),
            trajectory_xt=trajectory_xt if return_intermediates else None,
            trajectory_x0=trajectory_x0 if return_intermediates else None,
            timesteps=timesteps_list if return_intermediates else None
        )

    @torch.no_grad()
    def _sample_batch_heun(
        self,
        *,
        batch_size: int,
        generator: Optional[torch.Generator],
        return_intermediates: bool,
        sigma_min: float,
        sigma_max: float,
        rho: float,
        s_churn: float,
        s_min: float,
        s_max: float,
        s_noise: float,
    ) -> SamplingOutput:
        """EDM (Karras et al. 2022) deterministic Heun sampler over the rho-spaced schedule.

        Integrates the probability-flow ODE from ``sigma_max`` down to 0 with a 2nd-order Heun
        corrector (~``2 * num_steps - 1`` denoiser calls). ``s_churn`` > 0 enables the optional
        stochastic variant (Algorithm 2). Uses :meth:`denoise_sigma`, so it works for any model.
        """
        device = self.device
        dtype = torch.float64
        num_steps = self.num_steps

        # A model may restrict the usable sigma range (e.g. a network's sigma_min/sigma_max).
        sigma_min = max(float(sigma_min), float(getattr(self, "sampler_sigma_min", sigma_min)))
        sigma_max = min(float(sigma_max), float(getattr(self, "sampler_sigma_max", sigma_max)))

        # rho-spaced noise schedule from sigma_max down to sigma_min, with a final 0 appended.
        step_indices = torch.arange(num_steps, dtype=dtype, device=device)
        t_steps = (
            sigma_max ** (1.0 / rho)
            + step_indices / (num_steps - 1)
            * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
        ) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        shape = (batch_size, self.n_channels, self.resolution, self.resolution)
        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        x_next = latents * t_steps[0]  # start at the top noise scale sigma_max

        trajectory_xt: Optional[List[torch.Tensor]] = [] if return_intermediates else None
        trajectory_x0: Optional[List[torch.Tensor]] = [] if return_intermediates else None
        timesteps_list: Optional[List[int]] = [] if return_intermediates else None

        gamma_cap = 2.0 ** 0.5 - 1.0
        timesteps_iter = tqdm(range(num_steps), total=num_steps, desc="Sampling (Heun)", unit="step")
        for i in timesteps_iter:
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            x_cur = x_next

            # Optional stochastic churn: momentarily raise the noise level t_cur -> t_hat.
            gamma = min(s_churn / num_steps, gamma_cap) if (s_min <= t_cur <= s_max) else 0.0
            t_hat = t_cur + gamma * t_cur
            if gamma > 0:
                noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
                x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).clamp(min=0).sqrt() * s_noise * noise
            else:
                x_hat = x_cur

            # Euler predictor: slope dx/dsigma = (x - D(x, sigma)) / sigma.
            denoised = self.denoise_sigma(x_hat.to(torch.float32), t_hat).to(dtype)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Heun corrector: re-evaluate at the predicted point and average (skip final step).
            if i < num_steps - 1:
                denoised_next = self.denoise_sigma(x_next.to(torch.float32), t_next).to(dtype)
                d_prime = (x_next - denoised_next) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            if return_intermediates:
                if trajectory_xt is not None:
                    trajectory_xt.append(x_next.detach().float().cpu())
                if trajectory_x0 is not None:
                    trajectory_x0.append(denoised.detach().float().cpu())
                if timesteps_list is not None:
                    timesteps_list.append(i)

        return SamplingOutput(
            images=x_next.detach().float().cpu(),
            trajectory_xt=trajectory_xt if return_intermediates else None,
            trajectory_x0=trajectory_x0 if return_intermediates else None,
            timesteps=timesteps_list if return_intermediates else None,
        )