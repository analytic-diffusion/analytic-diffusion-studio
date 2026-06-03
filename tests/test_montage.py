"""Tests for the comparison montage builder (hermetic, synthetic grids)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# make_comparison_montage.py lives at the repo root, not inside the package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from make_comparison_montage import build_montage, _sort_key  # noqa: E402


def _fake_grid(path: Path, w: int = 128, h: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_sort_order_prefers_canonical():
    keys = sorted(["optimal", "wiener", "edm_unet"], key=_sort_key)
    assert keys == ["edm_unet", "wiener", "optimal"]
    # Unknown models sort last but stay deterministic.
    keys2 = sorted(["zzz", "edm_unet"], key=_sort_key)
    assert keys2 == ["edm_unet", "zzz"]


def test_build_montage_stacks_all_rows(tmp_path):
    runs = tmp_path / "runs"
    group = runs / "comparison_demo"
    for m in ("edm_unet", "wiener", "optimal"):
        _fake_grid(group / f"cmp_demo_{m}" / "grid.png")

    out = build_montage("demo", runs, scale=2)

    assert out == group / "montage.png"
    assert out.exists()
    img = Image.open(out)
    # At least the upscaled grid width and three stacked rows tall.
    assert img.size[0] >= 128 * 2
    assert img.size[1] >= 3 * (32 * 2)


def test_build_montage_skips_runs_without_grid(tmp_path):
    runs = tmp_path / "runs"
    group = runs / "comparison_demo"
    _fake_grid(group / "cmp_demo_edm_unet" / "grid.png")
    (group / "cmp_demo_wiener").mkdir(parents=True)  # no grid.png -> skipped

    out = build_montage("demo", runs, scale=1)
    assert out.exists()


def test_missing_group_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_montage("does_not_exist", tmp_path, scale=1)
