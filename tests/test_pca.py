"""Tests for precomputed-PCA download/convert utilities (hermetic, no network)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from local_diffusion.utils.pca import (
    PCA_FILENAMES,
    PIXEL_SCALE,
    PIXEL_SHIFT,
    _convert_pca_to_wiener,
    precomputed_pca_available,
    resolve_wiener_components,
)
from local_diffusion.utils.wiener import save_wiener_filter


def _synthetic_pca(d: int = 16, n: int = 200, seed: int = 0):
    """Build a {eigval, eigvec, imgmean} PCA dict from random [0, 1] data."""
    g = torch.Generator().manual_seed(seed)
    x01 = torch.rand(n, d, generator=g) * 0.4 + 0.3  # roughly natural-image-ish range
    cov01 = torch.cov(x01.T)
    eigval, eigvec = torch.linalg.eigh(cov01)  # ascending, unsorted relative to svd
    return {"eigval": eigval, "eigvec": eigvec, "imgmean": x01.mean(0)}, x01, cov01


def test_convert_mapping_shapes_and_scaling():
    pca, x01, cov01 = _synthetic_pca()
    U, LA, Vh, mean = _convert_pca_to_wiener(pca)

    d = pca["eigval"].numel()
    assert U.shape == (d, d)
    assert LA.shape == (d,)
    assert Vh.shape == (d, d)
    assert mean.shape == (d,)

    # Vh must be U^T (symmetric PSD covariance) and eigenvalues sorted descending.
    assert torch.allclose(Vh, U.T, atol=1e-6)
    assert torch.all(LA[:-1] >= LA[1:])

    # Mean and eigenvalues rescaled from [0, 1] into [-1, 1] coordinates.
    expected_mean = PIXEL_SCALE * pca["imgmean"] + PIXEL_SHIFT
    assert torch.allclose(mean, expected_mean, atol=1e-6)
    expected_top = (PIXEL_SCALE ** 2) * pca["eigval"].max()
    assert torch.allclose(LA[0], expected_top, atol=1e-5)


def test_convert_reconstructs_rescaled_covariance():
    pca, x01, cov01 = _synthetic_pca()
    U, LA, Vh, mean = _convert_pca_to_wiener(pca)

    S = U @ torch.diag(LA) @ Vh
    # S should be symmetric and equal 4 * cov01 (covariance of 2*x01 - 1).
    assert torch.allclose(S, S.T, atol=1e-5)
    assert torch.allclose(S, (PIXEL_SCALE ** 2) * cov01, atol=1e-4)


def test_precomputed_pca_available_keys():
    assert precomputed_pca_available("cifar10", 32)
    assert precomputed_pca_available("ffhq", 64)
    assert precomputed_pca_available("imagenet", 64)
    assert precomputed_pca_available("AFHQV2", 64)  # case-insensitive
    assert not precomputed_pca_available("mnist", 28)
    assert not precomputed_pca_available("ffhq", 32)  # wrong resolution
    # Every registered filename ends with the expected suffix.
    assert all(v.endswith("_PCA.pt") for v in PCA_FILENAMES.values())


def test_resolve_wiener_components_uses_cache(tmp_path):
    """When components are cached, resolve must return them without touching network."""
    d = 12
    U = torch.linalg.qr(torch.randn(d, d))[0]
    LA = torch.linspace(5.0, 0.1, d)
    Vh = U.T.contiguous()
    mean = torch.randn(d)
    save_wiener_filter(U, LA, Vh, mean, tmp_path)

    # Dummy dataset/dataloader that would error if iterated -- proves no recompute happens.
    dataset = SimpleNamespace(name="cifar10", resolution=32, dataloader=None)
    rU, rLA, rVh, rmean = resolve_wiener_components(
        tmp_path, dataset, device=torch.device("cpu"), n_channels=3,
        use_precomputed_pca=False,
    )
    assert torch.allclose(rU, U)
    assert torch.allclose(rLA, LA)
    assert torch.allclose(rVh, Vh)
    assert torch.allclose(rmean, mean)
