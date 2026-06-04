import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from local_diffusion.data import DatasetBundle
from local_diffusion.models.base import BaseDenoiser
from local_diffusion.utils import default_wiener_path, resolve_wiener_components
from local_diffusion.models import register_model


LOGGER = logging.getLogger(__name__)


@register_model("wiener")
class DenoisingWiener(BaseDenoiser):
    def __init__(
        self,
        dataset: DatasetBundle,
        device: str,
        num_steps: int,
        *,
        params: Optional[Dict[str, object]] = None,
        **kwargs,
    ):
        params = params or {}
        super().__init__(
            resolution=dataset.resolution,
            device=device,
            num_steps=num_steps,
            in_channels=dataset.in_channels,
            dataset_name=dataset.name,
            **kwargs,
        )
        
        # Allow precomputed PCA download (when available) to skip covariance + SVD.
        self.use_precomputed_pca = bool(params.get("use_precomputed_pca", True))

        wiener_path = params.get("wiener_path", None)
        self.wiener_path = Path(wiener_path) if wiener_path else default_wiener_path(dataset)

    def train(self, dataset: DatasetBundle):  # type: ignore[override]
        """Load, download (precomputed PCA), or compute Wiener filter matrices."""

        U, LA, Vh, mean = resolve_wiener_components(
            self.wiener_path,
            dataset,
            device=self.device,
            n_channels=self.n_channels,
            use_precomputed_pca=self.use_precomputed_pca,
        )

        self.register_buffer("U", U)
        self.register_buffer("LA", LA)
        self.register_buffer("Vh", Vh)
        self.register_buffer("mean", mean.to(self.device))
        return self

    def _shrink_factors(self, timestep: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-PCA-direction Wiener shrinkage and sqrt(alpha_bar) for a timestep."""
        if not all(hasattr(self, attr) for attr in ["U", "LA", "Vh", "mean"]):
            raise RuntimeError(
                "Model not trained. Call model.train(dataset) before sampling."
            )
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep].to(self.LA.device)
        beta_prod_t = 1 - alpha_prod_t
        # s_k = alpha_bar * lambda_k / (beta_bar + alpha_bar * lambda_k)  in [0, 1]
        shrink = alpha_prod_t * self.LA / (beta_prod_t + alpha_prod_t * self.LA)
        return shrink, alpha_prod_t.sqrt()

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

        timestep_index = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        shrink, sqrt_alpha = self._shrink_factors(timestep_index)

        # The Wiener estimate is  x0 = mean + U diag(s) U^T (x_t / sqrt(alpha_bar) - mean).
        # Apply it directly in the PCA basis (project -> shrink -> reconstruct) instead of
        # materializing the [n, n] filter: this is O(n * k) per step with no n x n matrix.
        x = latents.flatten(start_dim=1)                 # [batch, n]
        residual = x / sqrt_alpha - self.mean.unsqueeze(0)  # [batch, n]
        coeff = residual @ self.U                          # [batch, k]  project onto eigenvectors
        coeff = coeff * shrink.unsqueeze(0)                # shrink each direction
        pred_x0 = self.mean.unsqueeze(0) + coeff @ self.Vh  # [batch, n]  reconstruct (Vh = U^T)

        return pred_x0.view_as(latents)
