"""EDM (Karras et al. 2022) pretrained UNet as a denoiser in this framework.

Wraps an NVLabs EDM ``EDMPrecond`` network (Song/Dhariwal UNet) so it plugs into the
repo's DDIM sampling loop. The network is loaded from:

* a local checkpoint (``params.checkpoint_path``) -- official ``.pkl`` or converted ``.pt``;
* a converted ``.pt`` re-hosted on HuggingFace (``params.hf_repo`` + ``params.hf_filename``);
* otherwise the official ``.pkl`` from the NVIDIA EDM CDN, chosen by dataset name.

No clone of the upstream ``edm`` repo is required -- a minimal subset is vendored under
``local_diffusion.external.edm`` (see that package for licensing/attribution).

Sigma bridge (VP). The sampler hands us a DDPM-scaled latent ``x_t`` and an integer
timestep with cumulative ``alpha_bar``. EDM's denoiser ``D(x; sigma)`` expects a
unit-scaled input ``x = x0 + sigma * eps``. Since
``x_t = sqrt(alpha_bar) x0 + sqrt(1 - alpha_bar) eps``::

    sigma  = sqrt((1 - alpha_bar) / alpha_bar)
    x_edm  = x_t / sqrt(alpha_bar)            # == x0 + sigma * eps
    pred_x0 = D(x_edm, sigma)

This is schedule-agnostic: we feed the latent's true noise level, so any well-trained
sigma-conditioned denoiser denoises correctly regardless of the sampler's beta schedule.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from local_diffusion.data import DatasetBundle
from local_diffusion.models import register_model
from local_diffusion.models.base import BaseDenoiser
from local_diffusion.external import edm as edm_vendor

LOGGER = logging.getLogger(__name__)


# Official EDM VP checkpoints, keyed by this repo's dataset names (and a few aliases).
EDM_CDN_BASE = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained"
EDM_CDN_URLS = {
    "cifar10": f"{EDM_CDN_BASE}/edm-cifar10-32x32-uncond-vp.pkl",
    "ffhq": f"{EDM_CDN_BASE}/edm-ffhq-64x64-uncond-vp.pkl",
    "ffhq64": f"{EDM_CDN_BASE}/edm-ffhq-64x64-uncond-vp.pkl",
    "afhq": f"{EDM_CDN_BASE}/edm-afhqv2-64x64-uncond-vp.pkl",
    "afhqv2": f"{EDM_CDN_BASE}/edm-afhqv2-64x64-uncond-vp.pkl",
}

_DEFAULT_CACHE = Path("data/models/edm")


def _download(url: str, dest_dir: Path) -> Path:
    """Download ``url`` into ``dest_dir`` (reused if already present)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(url.split("?")[0]).name
    if dest.exists():
        LOGGER.info("Using cached EDM checkpoint at %s", dest)
        return dest
    LOGGER.info("Downloading EDM checkpoint %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    return dest


@register_model("edm_unet")
class EDMUNet(BaseDenoiser):
    """Pretrained EDM UNet denoiser wrapped for the DDIM sampler."""

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

        self.use_fp16 = bool(params.get("use_fp16", False))
        self.eps = 1e-8

        ckpt_path = self._resolve_checkpoint(params, dataset.name)
        self.net = self._load_network(ckpt_path, params)
        self.net = self.net.to(self.device).eval().requires_grad_(False)

        # Expose the network's supported sigma range to the Heun sampler.
        self.sampler_sigma_min = float(getattr(self.net, "sigma_min", 0.0))
        self.sampler_sigma_max = float(getattr(self.net, "sigma_max", float("inf")))

        # Warn (don't fail) on an obvious dataset/architecture mismatch.
        if getattr(self.net, "img_resolution", self.resolution) != self.resolution:
            LOGGER.warning(
                "EDM checkpoint resolution %s != dataset resolution %s",
                self.net.img_resolution,
                self.resolution,
            )
        if getattr(self.net, "img_channels", self.n_channels) != self.n_channels:
            LOGGER.warning(
                "EDM checkpoint channels %s != dataset channels %s",
                self.net.img_channels,
                self.n_channels,
            )

    # ----- checkpoint resolution / loading -------------------------------------------

    def _resolve_checkpoint(self, params: Dict[str, object], dataset_name: str) -> Path:
        cache_dir = Path(str(params.get("cache_dir", _DEFAULT_CACHE)))

        local = params.get("checkpoint_path")
        if local:
            path = Path(str(local))
            if not path.exists():
                raise FileNotFoundError(f"EDM checkpoint not found: {path}")
            return path

        # Converted checkpoint re-hosted on HuggingFace.
        hf_repo = params.get("hf_repo")
        hf_filename = params.get("hf_filename")
        if hf_repo and hf_filename:
            from local_diffusion.utils import hf_download

            return hf_download(
                str(hf_repo), str(hf_filename),
                repo_type=str(params.get("hf_repo_type", "model")),
                dest_dir=cache_dir,
            )

        url = params.get("checkpoint_url") or EDM_CDN_URLS.get(dataset_name.lower())
        if not url:
            raise ValueError(
                f"No EDM checkpoint for dataset '{dataset_name}'. Set "
                f"model.params.checkpoint_path / checkpoint_url / hf_repo+hf_filename, "
                f"or use one of {sorted(EDM_CDN_URLS)}."
            )
        return _download(str(url), cache_dir)

    def _load_network(self, path: Path, params: Dict[str, object]):
        suffix = path.suffix.lower()
        if suffix in (".pt", ".pth"):
            blob = torch.load(path, map_location="cpu", weights_only=False)
            if not (isinstance(blob, dict) and "init_kwargs" in blob and "state_dict" in blob):
                raise ValueError(
                    f"Converted EDM checkpoint {path} must contain 'init_kwargs' and "
                    f"'state_dict' (produced by convert_edm_checkpoint.py)."
                )
            net = edm_vendor.build_edm_network(dict(blob["init_kwargs"]))
            net.load_state_dict(blob["state_dict"])
            LOGGER.info("Loaded converted EDM checkpoint from %s", path)
            return net

        # Official EDM pickle.
        net = edm_vendor.load_edm_pickle(path, key=str(params.get("pickle_key", "ema")))
        LOGGER.info("Loaded EDM pickle checkpoint from %s", path)
        return net

    # ----- denoiser API ---------------------------------------------------------------

    def train(self, dataset: DatasetBundle):  # type: ignore[override]
        # Pretrained network; nothing to fit.
        return self

    @torch.no_grad()
    def denoise_sigma(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Exact EDM denoiser ``D(x; sigma)`` (overrides the VP bridge for the Heun sampler)."""
        sigma_b = torch.as_tensor(sigma, dtype=torch.float32, device=x.device)
        if sigma_b.ndim == 0:
            sigma_b = sigma_b.expand(x.shape[0])
        pred_x0 = self.net(x.to(torch.float32), sigma_b, force_fp32=not self.use_fp16)
        return pred_x0.to(x.dtype)

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

        t = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        alpha_prod_t = self.scheduler.alphas_cumprod[t].to(latents.device)
        sqrt_alpha = alpha_prod_t.clamp(min=self.eps).sqrt()
        sigma = ((1.0 - alpha_prod_t) / alpha_prod_t.clamp(min=self.eps)).sqrt()

        # Map the DDPM latent to EDM's unit-scaled input, then reuse the exact denoiser.
        x_edm = latents / sqrt_alpha
        return self.denoise_sigma(x_edm, sigma)
