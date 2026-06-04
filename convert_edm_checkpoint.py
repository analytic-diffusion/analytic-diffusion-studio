"""Convert an official EDM ``.pkl`` into a dependency-light ``.pt`` checkpoint.

The official NVLabs EDM checkpoints are pickled ``nn.Module`` objects that require the
EDM ``torch_utils`` / ``dnnlib`` machinery to unpickle. This script reads one (using the
vendored EDM subset -- no upstream clone needed) and re-saves just the constructor
kwargs and weights as ``{"init_kwargs": ..., "state_dict": ..., "meta": ...}``. The
result loads in this repo via ``edm_unet`` with ``torch.load`` + ``load_state_dict`` and
no reliance on the EDM pickle/persistence layer -- handy for re-hosting and for
robustness across torch versions.

Usage::

    uv run convert_edm_checkpoint.py <input.pkl|url> <output.pt> [--key ema]

Example::

    uv run convert_edm_checkpoint.py \
        https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
        data/models/edm/edm-cifar10-32x32-uncond-vp.pt
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

import torch

from local_diffusion.external import edm as edm_vendor


def convert(source: str, output: str, key: str = "ema") -> None:
    src_path = source
    if source.startswith("http://") or source.startswith("https://"):
        cached = Path(tempfile.gettempdir()) / Path(source.split("?")[0]).name
        if not cached.exists():
            print(f"downloading {source} -> {cached}")
            urllib.request.urlretrieve(source, cached)
        src_path = str(cached)

    net = edm_vendor.load_edm_pickle(src_path, key=key)

    blob = {
        "init_kwargs": dict(net.init_kwargs),
        "state_dict": net.state_dict(),
        "meta": {
            "img_resolution": int(net.img_resolution),
            "img_channels": int(net.img_channels),
            "label_dim": int(net.label_dim),
            "sigma_min": float(net.sigma_min),
            "sigma_max": float(net.sigma_max),
            "sigma_data": float(net.sigma_data),
        },
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out)
    print(f"saved converted checkpoint -> {out} ({len(blob['state_dict'])} tensors)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Path or URL to an official EDM .pkl")
    parser.add_argument("output", help="Output path for the converted .pt")
    parser.add_argument("--key", default="ema", help="Snapshot key to extract (default: ema)")
    args = parser.parse_args()
    convert(args.source, args.output, key=args.key)


if __name__ == "__main__":
    main()
