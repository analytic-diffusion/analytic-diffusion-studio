import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from local_diffusion.data import DatasetBundle
from local_diffusion.models.base import BaseDenoiser
from local_diffusion.utils import resolve_wiener_components
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
        
        self.wiener_path = params.get("wiener_path", None)
        # Allow precomputed PCA download (when available) to skip covariance + SVD.
        self.use_precomputed_pca = bool(params.get("use_precomputed_pca", True))

        # If path not provided, default to data/models/wiener/<dataset>_<resolution>
        if self.wiener_path is None:
            default_root = Path("data/models/wiener")
            self.wiener_path = default_root / f"{dataset.name}_{dataset.resolution}"
        else:
            self.wiener_path = Path(self.wiener_path)

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

    def _get_Lt_Ht(self, timestep: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not all(hasattr(self, attr) for attr in ["U", "LA", "Vh", "mean"]):
            raise RuntimeError(
                "Model not trained. Call model.train(dataset) before sampling."
            )
        
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t

        shrink_factors = alpha_prod_t * self.LA / (beta_prod_t + alpha_prod_t * self.LA)
        LAshrink = torch.diag(shrink_factors)  # [n, n]
        LLt = self.U @ LAshrink @ self.Vh  # [n, n]

        I = torch.eye(LLt.shape[0], device=LLt.device)
        Ht = I - LLt
        Lt = LLt.clone() / torch.sqrt(alpha_prod_t)
        return Lt, Ht

    @torch.no_grad()
    def denoise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        del generator, kwargs
        
        if self.mean is None:
            raise RuntimeError(
                "Model not trained. Call model.train(dataset) before sampling."
            )

        timestep_index = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        Lt, Ht = self._get_Lt_Ht(timestep_index)

        # Flatten latents to [batch, n_pixels]
        latents_flat = latents.flatten(start_dim=1)  # [batch, n_pixels]
        
        # Apply Wiener filter: Lt @ x_t + Ht @ mean
        lx0_flat = (Lt @ latents_flat.T).T  # [n_pixels, n_pixels] @ [n_pixels, batch] -> [n_pixels, batch] -> [batch, n_pixels]
        mean_term_flat = (Ht @ self.mean.unsqueeze(-1)).squeeze(-1)  # [n_pixels, n_pixels] @ [n_pixels, 1] -> [n_pixels]
        
        # Combine and reshape back to image dimensions
        total_x0_flat = lx0_flat + mean_term_flat.unsqueeze(0)  # [batch, n_pixels]
        total_x0 = total_x0_flat.view_as(latents)

        return total_x0
