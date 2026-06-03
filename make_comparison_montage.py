"""Stitch the per-model sample grids of a comparison run into one labeled montage.

The denoiser comparison (``configs/comparison/``) runs each model separately, so each
``generate.py`` invocation only saves its own ``grid.png``. This script collects the
grids from a comparison run group and assembles a single side-by-side figure -- it is
invoked automatically at the end of ``run_comparison.sh``.

Usage::

    uv run make_comparison_montage.py <dataset> [--runs-dir data/runs] [--scale 3]

It reads ``<runs-dir>/comparison_<dataset>/cmp_*/grid.png`` and writes
``<runs-dir>/comparison_<dataset>/montage.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# Preferred row order and friendly labels (unknown models fall back to their key).
_ORDER = ["edm_unet", "wiener", "optimal", "scfdm", "pca_locality", "nearest_dataset", "baseline_unet"]
_LABELS = {
    "edm_unet": "edm_unet — pretrained EDM UNet",
    "wiener": "wiener — Gaussian / Wiener",
    "optimal": "optimal — empirical softmax",
    "scfdm": "scfdm — smoothed optimal",
    "pca_locality": "pca_locality — analytic locality",
    "nearest_dataset": "nearest_dataset — NN retrieval",
}

_LABEL_STRIP = 22  # px reserved above each row for its text label


def _model_key(run_dir: Path, dataset: str) -> str:
    name = run_dir.name
    prefix = f"cmp_{dataset}_"
    return name[len(prefix):] if name.startswith(prefix) else name


def _sort_key(key: str):
    return (_ORDER.index(key) if key in _ORDER else len(_ORDER), key)


def build_montage(dataset: str, runs_dir: Path, scale: int) -> Path:
    group_dir = runs_dir / f"comparison_{dataset}"
    if not group_dir.is_dir():
        raise FileNotFoundError(f"Comparison group not found: {group_dir}")

    runs = sorted(
        (d for d in group_dir.glob("cmp_*") if (d / "grid.png").exists()),
        key=lambda d: _sort_key(_model_key(d, dataset)),
    )
    if not runs:
        raise FileNotFoundError(f"No 'cmp_*/grid.png' grids found under {group_dir}")

    rows = []
    for run_dir in runs:
        key = _model_key(run_dir, dataset)
        label = _LABELS.get(key, key)
        grid = Image.open(run_dir / "grid.png").convert("RGB")
        w, h = grid.size
        big = grid.resize((w * scale, h * scale), Image.NEAREST)
        canvas = Image.new("RGB", (big.size[0], big.size[1] + _LABEL_STRIP), (255, 255, 255))
        canvas.paste(big, (0, _LABEL_STRIP))
        ImageDraw.Draw(canvas).text((4, 6), f"{label}", fill=(0, 0, 0))
        rows.append(np.asarray(canvas))

    width = max(r.shape[1] for r in rows)
    rows = [
        np.pad(r, ((0, 8), (0, width - r.shape[1]), (0, 0)), constant_values=255)
        for r in rows
    ]
    montage = Image.fromarray(np.concatenate(rows, axis=0))

    out_path = group_dir / "montage.png"
    montage.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset name (e.g. afhqv2, ffhq, cifar10)")
    parser.add_argument("--runs-dir", default="data/runs", help="Root runs directory")
    parser.add_argument("--scale", type=int, default=3, help="Nearest-neighbor upscale factor")
    args = parser.parse_args()

    out = build_montage(args.dataset, Path(args.runs_dir), args.scale)
    n_rows = len(list((Path(args.runs_dir) / f"comparison_{args.dataset}").glob("cmp_*/grid.png")))
    print(f"Saved comparison montage ({n_rows} models) -> {out}")


if __name__ == "__main__":
    main()
