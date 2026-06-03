"""Tests for the edm_unet model and the vendored EDM subset.

Hermetic by default: a tiny EDM network is built in-process and saved as a converted
checkpoint -- no 200 MB download. The real-checkpoint load is gated behind
RUN_NETWORK_TESTS=1.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from local_diffusion.external import edm as edm_vendor
from local_diffusion.models import create_model
from local_diffusion.models.edm_unet import EDM_CDN_URLS


TINY_KWARGS = dict(
    img_resolution=8, img_channels=3, label_dim=0,
    model_type="SongUNet", model_channels=4, channel_mult=[1, 1],
    num_blocks=1, attn_resolutions=[], dropout=0.0,
    embedding_type="positional", encoder_type="standard",
    channel_mult_noise=1, resample_filter=[1, 1], augment_dim=0,
)


def _make_converted_checkpoint(path):
    net = edm_vendor.build_edm_network(dict(TINY_KWARGS))
    torch.save(
        {
            "init_kwargs": dict(TINY_KWARGS),
            "state_dict": net.state_dict(),
            "meta": {"img_resolution": 8, "img_channels": 3, "sigma_data": 0.5},
        },
        path,
    )
    return net


def _tiny_dataset():
    return SimpleNamespace(resolution=8, in_channels=3, name="cifar10")


def test_vendored_build_network():
    net = edm_vendor.build_edm_network(dict(TINY_KWARGS))
    assert net.img_resolution == 8
    assert net.img_channels == 3
    out = net(torch.randn(2, 3, 8, 8), torch.full((2,), 0.7))
    assert out.shape == (2, 3, 8, 8)
    assert torch.isfinite(out).all()


def test_cdn_url_map_has_expected_datasets():
    for name in ("cifar10", "ffhq", "afhq", "afhqv2"):
        assert name in EDM_CDN_URLS
        assert EDM_CDN_URLS[name].endswith(".pkl")


def test_converted_checkpoint_load_and_denoise(tmp_path):
    ckpt = tmp_path / "tiny_edm.pt"
    ref_net = _make_converted_checkpoint(ckpt)

    model = create_model(
        "edm_unet", dataset=_tiny_dataset(), device="cpu", num_steps=4,
        params={"checkpoint_path": str(ckpt)},
    )

    latents = torch.randn(2, 3, 8, 8)
    pred = model.denoise(latents, torch.tensor(440))
    assert pred.shape == (2, 3, 8, 8)
    assert torch.isfinite(pred).all()

    # Loaded weights must match the reference net exactly.
    ref_net = ref_net.eval()
    xt = torch.randn(2, 3, 8, 8)
    s = torch.full((2,), 0.5)
    assert torch.allclose(model.net(xt, s), ref_net(xt, s), atol=1e-6)


def test_sigma_bridge_math(tmp_path):
    ckpt = tmp_path / "tiny_edm.pt"
    _make_converted_checkpoint(ckpt)
    model = create_model(
        "edm_unet", dataset=_tiny_dataset(), device="cpu", num_steps=4,
        params={"checkpoint_path": str(ckpt)},
    )

    seen = {}

    class Recorder(torch.nn.Module):
        def forward(self, x, sigma, **kwargs):
            seen["x"] = x.clone()
            seen["sigma"] = sigma.clone()
            return torch.zeros_like(x)

    model.net = Recorder()

    t = 500
    latents = torch.randn(2, 3, 8, 8)
    model.denoise(latents, torch.tensor(t))

    alpha_bar = model.scheduler.alphas_cumprod[t]
    expected_sigma = ((1.0 - alpha_bar) / alpha_bar).sqrt()
    expected_x = latents / alpha_bar.sqrt()

    assert torch.allclose(seen["sigma"], expected_sigma.expand(2).float(), atol=1e-6)
    assert torch.allclose(seen["x"], expected_x.float(), atol=1e-5)


@pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1",
    reason="downloads the real EDM checkpoint; set RUN_NETWORK_TESTS=1 to run",
)
def test_real_cifar_pkl_load():
    model = create_model(
        "edm_unet", dataset=SimpleNamespace(resolution=32, in_channels=3, name="cifar10"),
        device="cpu", num_steps=4, params={},
    )
    pred = model.denoise(torch.randn(2, 3, 32, 32), torch.tensor(440))
    assert pred.shape == (2, 3, 32, 32)
    assert torch.isfinite(pred).all()
