"""Tests for the sampling methods (DDIM dispatch + EDM Heun), fully hermetic.

A tiny EDM network is built in-process so the Heun integrator can be checked against a
reference implementation without any download.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from local_diffusion.external import edm as edm_vendor
from local_diffusion.models import create_model
from local_diffusion.models.base import BaseDenoiser


TINY_KWARGS = dict(
    img_resolution=8, img_channels=3, label_dim=0,
    model_type="SongUNet", model_channels=4, channel_mult=[1, 1],
    num_blocks=1, attn_resolutions=[], dropout=0.0,
    embedding_type="positional", encoder_type="standard",
    channel_mult_noise=1, resample_filter=[1, 1], augment_dim=0,
)


@pytest.fixture
def tiny_edm_model(tmp_path):
    net = edm_vendor.build_edm_network(dict(TINY_KWARGS))
    ckpt = tmp_path / "tiny_edm.pt"
    torch.save({"init_kwargs": dict(TINY_KWARGS), "state_dict": net.state_dict(), "meta": {}}, ckpt)
    return create_model(
        "edm_unet", dataset=SimpleNamespace(resolution=8, in_channels=3, name="cifar10"),
        device="cpu", num_steps=6, params={"checkpoint_path": str(ckpt)},
    )


def _reference_heun(net, latents, num_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    sigma_min = max(sigma_min, float(net.sigma_min))
    sigma_max = min(sigma_max, float(net.sigma_max))
    si = torch.arange(num_steps, dtype=torch.float64)
    t_steps = (sigma_max ** (1 / rho) + si / (num_steps - 1)
               * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    x = latents.to(torch.float64) * t_steps[0]
    for i, (tc, tn) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        d = net(x.float(), tc.float().expand(x.shape[0])).to(torch.float64)
        d_cur = (x - d) / tc
        x2 = x + (tn - tc) * d_cur
        if i < num_steps - 1:
            d2 = net(x2.float(), tn.float().expand(x.shape[0])).to(torch.float64)
            x2 = x + (tn - tc) * (0.5 * d_cur + 0.5 * (x2 - d2) / tn)
        x = x2
    return x


def test_heun_matches_reference(tiny_edm_model):
    shape = (2, 3, 8, 8)
    g = torch.Generator().manual_seed(0)
    out = tiny_edm_model.sample(num_samples=2, batch_size=2, generator=g, method="heun")

    g2 = torch.Generator().manual_seed(0)
    latents = torch.randn(shape, generator=g2, dtype=torch.float64)
    ref = _reference_heun(tiny_edm_model.net, latents, num_steps=tiny_edm_model.num_steps)

    assert torch.allclose(out.images.double(), ref, atol=1e-10)


def test_heun_nfe_skips_corrector_on_last_step(tiny_edm_model):
    calls = {"n": 0}
    original = tiny_edm_model.denoise_sigma

    def counting(x, sigma):
        calls["n"] += 1
        return original(x, sigma)

    tiny_edm_model.denoise_sigma = counting
    tiny_edm_model.sample(num_samples=1, batch_size=1, method="heun")

    # Heun: 2 evals per step, minus the corrector skipped on the final step.
    assert calls["n"] == 2 * tiny_edm_model.num_steps - 1


def test_method_dispatch_and_unknown(tiny_edm_model):
    ddim = tiny_edm_model.sample(num_samples=1, batch_size=1, method="ddim")
    heun = tiny_edm_model.sample(num_samples=1, batch_size=1, method="heun")
    assert ddim.images.shape == (1, 3, 8, 8)
    assert heun.images.shape == (1, 3, 8, 8)
    with pytest.raises(ValueError):
        tiny_edm_model.sample(num_samples=1, batch_size=1, method="euler_xyz")


def test_denoise_sigma_vp_bridge_math(tiny_edm_model):
    """The base (non-overridden) denoise_sigma must map sigma -> nearest VP timestep."""
    seen = {}

    def recorder(x_t, timestep):
        seen["x_t"] = x_t.clone()
        seen["t"] = int(timestep)
        return torch.zeros_like(x_t)

    tiny_edm_model.denoise = recorder
    x = torch.randn(2, 3, 8, 8)
    sigma = 0.5

    BaseDenoiser.denoise_sigma(tiny_edm_model, x, torch.tensor(sigma))

    alpha_bar = 1.0 / (1.0 + sigma ** 2)
    ac = tiny_edm_model.scheduler.alphas_cumprod
    expected_t = int(torch.argmin((ac - alpha_bar).abs()).item())
    assert seen["t"] == expected_t
    assert torch.allclose(seen["x_t"], x * (alpha_bar ** 0.5), atol=1e-6)
