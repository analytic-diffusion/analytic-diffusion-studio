"""Tests for the PCA-basis Wiener denoiser and the Heun step guard (hermetic)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from local_diffusion.models import create_model


def _wiener_with_synthetic_components(n_side=4, channels=3, num_steps=4, seed=0):
    n = channels * n_side * n_side
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(n, n, generator=g))[0]          # orthonormal eigvecs
    LA = torch.linspace(5.0, 0.01, n)                                # descending eigenvalues
    mean = torch.randn(n, generator=g)

    ds = SimpleNamespace(resolution=n_side, in_channels=channels, name="synthetic")
    model = create_model("wiener", dataset=ds, device="cpu", num_steps=num_steps, params={})
    model.register_buffer("U", Q)
    model.register_buffer("LA", LA)
    model.register_buffer("Vh", Q.t().contiguous())
    model.register_buffer("mean", mean)
    return model, Q, LA, mean, g


def test_pca_basis_matches_explicit_filter():
    model, Q, LA, mean, g = _wiener_with_synthetic_components()
    n = Q.shape[0]
    x = torch.randn(2, 3, 4, 4, generator=g)

    for t in (0, 1, 250, 999):
        alpha_bar = model.scheduler.alphas_cumprod[t]
        beta_bar = 1 - alpha_bar
        s = alpha_bar * LA / (beta_bar + alpha_bar * LA)
        LLt = Q @ torch.diag(s) @ Q.t()                              # explicit [n, n] filter

        x_edm = x.flatten(1) / alpha_bar.sqrt()
        expected = (mean.unsqueeze(0) + (x_edm - mean.unsqueeze(0)) @ LLt.t()).view_as(x)

        got = model.denoise(x, torch.tensor(t))
        assert torch.allclose(got, expected, atol=1e-5), f"mismatch at t={t}"


def test_denoise_before_train_raises():
    ds = SimpleNamespace(resolution=4, in_channels=3, name="synthetic")
    model = create_model("wiener", dataset=ds, device="cpu", num_steps=4, params={})
    with pytest.raises(RuntimeError):
        model.denoise(torch.randn(1, 3, 4, 4), torch.tensor(0))


def test_heun_requires_two_steps():
    model, *_ = _wiener_with_synthetic_components(num_steps=1)
    with pytest.raises(ValueError):
        model.sample(num_samples=1, batch_size=1, method="heun")
