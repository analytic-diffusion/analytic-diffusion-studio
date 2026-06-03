"""Download and convert precomputed PCA decompositions into Wiener SVD components.

The precomputed PCAs live in the HuggingFace dataset repo
``binxu/image_datasets_PCAs`` as ``<name>_PCA.pt`` files. Each file is a dict::

    {"eigval": [D], "eigvec": [D, D], "imgmean": [D]}

computed on images in the ``[0, 1]`` pixel range (``D = channels * res * res``).
The covariance is ``eigvec @ diag(eigval) @ eigvec.T`` and ``imgmean`` is the per-pixel
mean.

This framework operates in the ``[-1, 1]`` pixel range (see
``data.utils.compose_transform``), where ``x = 2 * x01 - 1``. Under that affine map the
mean becomes ``2 * mean - 1`` and the covariance scales by ``2 ** 2 = 4`` while the
eigenvectors are unchanged. We therefore rescale on import and store the result as the
Wiener SVD components ``(U, LA, Vh, mean)`` expected by
``utils.wiener.load_wiener_filter`` -- i.e. ``S = U @ diag(LA) @ Vh`` with ``Vh = U.T``
for the symmetric PSD covariance ``S``.

This lets both the ``wiener`` and ``pca_locality`` models reuse a precomputed
decomposition instead of recomputing the full covariance and its SVD, which is
prohibitive at higher resolutions (e.g. at 64x64x3 the covariance is 12288x12288).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

from .wiener import compute_wiener_filter, load_wiener_filter, save_wiener_filter

LOGGER = logging.getLogger(__name__)


# HuggingFace dataset repo hosting the precomputed PCA files.
PCA_HF_REPO = "binxu/image_datasets_PCAs"

# Map "<dataset_name>_<resolution>" -> PCA filename in the repo above.
# Aliases (e.g. afhq / afhqv2) point at the same file so either dataset name works.
PCA_FILENAMES = {
    "cifar10_32": "cifar32_PCA.pt",
    "afhq_64": "afhqv264_PCA.pt",
    "afhqv2_64": "afhqv264_PCA.pt",
    "ffhq_64": "ffhq64_PCA.pt",
    "imagenet_64": "imagenet64_PCA.pt",
}

# Affine map from the [0, 1] range the PCAs were computed in to the [-1, 1] range
# used throughout this framework: x = PIXEL_SCALE * x01 + PIXEL_SHIFT.
PIXEL_SCALE = 2.0
PIXEL_SHIFT = -1.0


def precomputed_pca_key(name: str, resolution: int) -> str:
    return f"{name.lower()}_{int(resolution)}"


def precomputed_pca_available(name: str, resolution: int) -> bool:
    """Return True if a precomputed PCA exists for this dataset + resolution."""
    return precomputed_pca_key(name, resolution) in PCA_FILENAMES


def _convert_pca_to_wiener(
    pca: dict,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a ``{eigval, eigvec, imgmean}`` PCA dict to Wiener SVD components.

    Returns ``(U, LA, Vh, mean)`` rescaled to the framework's ``[-1, 1]`` range and
    sorted by descending eigenvalue (matching ``torch.linalg.svd`` ordering).
    """
    eigval = pca["eigval"].float()
    eigvec = pca["eigvec"].float()
    imgmean = pca["imgmean"].float()

    # Sort by descending eigenvalue so the components match torch.linalg.svd's
    # convention, keeping eigenvectors aligned with their eigenvalues.
    order = torch.argsort(eigval, descending=True)
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    U = eigvec.contiguous()
    LA = (PIXEL_SCALE ** 2) * eigval
    Vh = eigvec.t().contiguous()
    mean = PIXEL_SCALE * imgmean + PIXEL_SHIFT

    if device is not None:
        U, LA, Vh, mean = U.to(device), LA.to(device), Vh.to(device), mean.to(device)
    return U, LA, Vh, mean


def download_precomputed_pca(
    name: str,
    resolution: int,
    save_path: Path,
    *,
    device: Optional[torch.device] = None,
    cache_dir: Optional[str] = None,
) -> bool:
    """Fetch and convert a precomputed PCA into the Wiener cache at ``save_path``.

    Parameters
    ----------
    name : str
        Dataset name (e.g. ``"ffhq"``, ``"cifar10"``).
    resolution : int
        Spatial resolution; combined with ``name`` to select the PCA file.
    save_path : Path
        Directory in which to write ``U.pt``, ``LA.pt``, ``Vh.pt`` and ``mean.pt``.
    device : torch.device, optional
        Device on which to place the converted tensors before saving.
    cache_dir : str, optional
        HuggingFace download cache directory.

    Returns
    -------
    bool
        True if a precomputed PCA was found, downloaded and converted; False if no
        precomputed PCA exists for this ``name`` + ``resolution``.
    """
    filename = PCA_FILENAMES.get(precomputed_pca_key(name, resolution))
    if filename is None:
        return False

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "huggingface_hub is required to download precomputed PCAs. "
            "Install it with `uv pip install huggingface_hub`."
        ) from exc

    LOGGER.info(
        "Downloading precomputed PCA '%s' for %s@%d from %s...",
        filename,
        name,
        resolution,
        PCA_HF_REPO,
    )
    local_path = hf_hub_download(
        repo_id=PCA_HF_REPO,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    pca = torch.load(local_path, map_location="cpu", weights_only=True)
    U, LA, Vh, mean = _convert_pca_to_wiener(pca, device=device)
    save_wiener_filter(U, LA, Vh, mean, Path(save_path))
    LOGGER.info("Converted precomputed PCA and saved Wiener components to %s", save_path)
    return True


def resolve_wiener_components(
    wiener_path: Path,
    dataset,
    device: torch.device,
    n_channels: int,
    *,
    use_precomputed_pca: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Wiener SVD components ``(U, LA, Vh, mean)`` for a dataset.

    Resolution order:
      1. Load cached components from ``wiener_path`` if present.
      2. Otherwise, if ``use_precomputed_pca`` and a precomputed PCA is published for
         this dataset + resolution, download/convert it into ``wiener_path``.
      3. Otherwise, compute the covariance from the dataloader, SVD it, and cache it.

    Shared by the ``wiener`` and ``pca_locality`` models.
    """
    wiener_path = Path(wiener_path)
    try:
        return load_wiener_filter(wiener_path, device=device)
    except FileNotFoundError:
        pass

    if use_precomputed_pca and download_precomputed_pca(
        dataset.name, dataset.resolution, wiener_path, device=device
    ):
        return load_wiener_filter(wiener_path, device=device)

    LOGGER.info(
        "No cached Wiener filter or precomputed PCA for %s@%d. Computing from dataset...",
        dataset.name,
        dataset.resolution,
    )
    S, mean = compute_wiener_filter(
        dataloader=dataset.dataloader,
        device=device,
        resolution=dataset.resolution,
        n_channels=n_channels,
    )
    U, LA, Vh = torch.linalg.svd(S)
    save_wiener_filter(U, LA, Vh, mean, wiener_path)
    return load_wiener_filter(wiener_path, device=device)
