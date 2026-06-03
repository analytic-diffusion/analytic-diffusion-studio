

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Dict, Any

import torch
from tqdm import tqdm

from local_diffusion.data import DatasetBundle
from local_diffusion.models import register_model
from local_diffusion.models.base import BaseDenoiser
from local_diffusion.utils import default_wiener_path, resolve_wiener_components

LOGGER = logging.getLogger(__name__)


class WeightedStreamingSoftmax:
    """
    Running weighted softmax averaging over images.
    See WSSM in https://arxiv.org/abs/2509.09672 Appendix C3 for more details.
    Crutial for efficient implementation of our analytical diffusion model.
    """

    def __init__(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.sum_weighted: Optional[torch.Tensor] = None  # [b, n]
        self.sum_weights: Optional[torch.Tensor] = None  # [b, n]
        self.eps = eps

    def add(self, x0b: torch.Tensor, logits: torch.Tensor) -> None:
        b, k, n = logits.shape
        expected_shape = (k, n)

        if x0b.shape != expected_shape:
            raise ValueError(f"Expected x0b of shape {expected_shape}, got {x0b.shape}")

        x0b = x0b.to(dtype=self.dtype, device=self.device)
        logits = logits.to(dtype=self.dtype, device=self.device)

        # Softmax over k in a numerically stable way
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits_exp = torch.exp(logits - logits_max)
        weights = logits_exp / logits_exp.sum(dim=1, keepdim=True)

        weighted_sum = torch.einsum("bkn,kn->bn", weights, x0b)
        weight_sum = weights.sum(dim=1)

        if self.sum_weighted is None:
            self.sum_weighted = weighted_sum
            self.sum_weights = weight_sum
        else:
            self.sum_weighted += weighted_sum
            self.sum_weights += weight_sum

    def get_average(self) -> Optional[torch.Tensor]:
        if self.sum_weighted is None or self.sum_weights is None:
            return None

        return self.sum_weighted / (self.sum_weights + self.eps)


@register_model("pca_locality")
class PCALocalityDenoiser(BaseDenoiser):
    """Our analytical diffusion model from https://arxiv.org/abs/2509.09672"""

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
            resolution=dataset.resolution,
            device=device,
            num_steps=num_steps,
            in_channels=dataset.in_channels,
            dataset_name=dataset.name,
            **kwargs,
        )

        self.temperature = float(params.get("temperature", 1.0))
        self.mask_threshold = float(params.get("mask_threshold", 0.02))
        self.eps = 1e-6

        # Allow precomputed PCA download (when available) to skip covariance + SVD.
        self.use_precomputed_pca = bool(params.get("use_precomputed_pca", True))

        # Share the Wiener cache path with the Wiener model.
        wiener_path = params.get("wiener_path", None)
        self.wiener_path = Path(wiener_path) if wiener_path else default_wiener_path(dataset)

        self.dataset: Optional[DatasetBundle] = None

    def train(self, dataset: DatasetBundle):  # type: ignore[override]
        """Load, download (precomputed PCA), or compute Wiener SVD; keep dataset ref."""
        U, LA, Vh, mean = resolve_wiener_components(
            self.wiener_path,
            dataset,
            device=self.device,
            n_channels=self.n_channels,
            use_precomputed_pca=self.use_precomputed_pca,
        )

        # Keep buffers on the target device to avoid per-call transfers
        self.register_buffer("U", U.to(self.device))
        self.register_buffer("LA", LA.to(self.device))
        self.register_buffer("Vh", Vh.to(self.device))
        self.register_buffer("mean", mean.to(self.device))
        self.dataset = dataset
        return self

    def _projection_mask(
        self, timestep_index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute LLᵀ mask and scheduler scalars for a timestep."""
        if not all(hasattr(self, attr) for attr in ["U", "LA", "Vh"]):
            raise RuntimeError("Model not trained. Call model.train(dataset) first.")

        alpha_prod_t = self.scheduler.alphas_cumprod[timestep_index].to(self.device)
        beta_prod_t = 1 - alpha_prod_t

        shrink_factors = alpha_prod_t * self.LA / (beta_prod_t + alpha_prod_t * self.LA)
        LAshrink = torch.diag(shrink_factors)
        LLt = self.U @ LAshrink @ self.Vh  # [n, n]

        # Normnilize the diagonal of the masks to 1
        denom = torch.diagonal(LLt).unsqueeze(1)
        denom[denom.abs() < self.eps] = 1.0
        mask = LLt / denom
        
        # Binarize the mask with the given threshold
        if self.mask_threshold > 0:
            threshold = mask.abs().max() * self.mask_threshold
            mask = torch.where(mask.abs() >= threshold, torch.ones_like(mask), torch.zeros_like(mask))

        return mask, alpha_prod_t, beta_prod_t

    @torch.no_grad()
    def denoise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del generator, kwargs

        if self.dataset is None:
            raise RuntimeError("Model not trained. Call model.train(dataset) first.")

        t_idx = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        mask, alpha_prod_t, beta_prod_t = self._projection_mask(t_idx)
        sqrt_alpha = torch.sqrt(alpha_prod_t).to(latents.device, dtype=latents.dtype)

        xt = latents.flatten(start_dim=1)  # [b, n]
        first_moment = WeightedStreamingSoftmax(device=latents.device, dtype=latents.dtype)

        for x0_batch in tqdm(self.dataset.dataloader, desc="PCA locality", leave=False):
            images = x0_batch[0] if isinstance(x0_batch, (tuple, list)) else x0_batch
            x0b = images.to(latents.device, dtype=latents.dtype).flatten(start_dim=1)  # [k, n]

            delta = (xt.unsqueeze(1) - sqrt_alpha * x0b.unsqueeze(0)) ** 2  # [b, k, n]
            ds_chunk = torch.einsum("bkn,nm->bkm", delta, mask)  # [b, k, n]
            logits = -ds_chunk / (2 * beta_prod_t * self.temperature)

            first_moment.add(x0b, logits)

        x0_mean = first_moment.get_average()
        if x0_mean is None:
            raise RuntimeError("Failed to compute softmax average for PCA locality.")

        pred_x0 = x0_mean.view_as(latents)
        return pred_x0