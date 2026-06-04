"""Tests for the EDM-format zip image dataset loader (hermetic, no network)."""

from __future__ import annotations

import json
import pickle
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from local_diffusion.data import utils as dutils
from local_diffusion.data.datasets import DATASET_REGISTRY
from local_diffusion.data.hf_datasets import EDMZipImageDataset


N_FRAMES = 7
RES = 8


@pytest.fixture
def edm_zip(tmp_path):
    """Create a tiny EDM-layout zip: nested PNG frames + a dataset.json sidecar."""
    zip_path = tmp_path / "tiny-8x8.zip"
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i in range(N_FRAMES):
            arr = rng.integers(0, 256, size=(RES, RES, 3), dtype=np.uint8)
            from io import BytesIO

            buf = BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            # EDM nests frames under shard folders, e.g. 00000/img00000000.png
            zf.writestr(f"00000/img{i:08d}.png", buf.getvalue())
        zf.writestr("dataset.json", json.dumps({"labels": None}))
    return zip_path


def test_reads_frames_in_normalized_range(edm_zip):
    transform = dutils.compose_transform(RES, in_channels=3)
    ds = EDMZipImageDataset(zip_path=str(edm_zip), transform=transform)

    assert len(ds) == N_FRAMES  # dataset.json must be ignored
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, RES, RES)
    assert label == 0
    assert img.min() >= -1.0 - 1e-6 and img.max() <= 1.0 + 1e-6


def test_frames_sorted(edm_zip):
    ds = EDMZipImageDataset(zip_path=str(edm_zip))
    assert ds._image_fnames == sorted(ds._image_fnames)
    assert all(name.endswith(".png") for name in ds._image_fnames)


def test_dataloader_batches(edm_zip):
    transform = dutils.compose_transform(RES, in_channels=3)
    ds = EDMZipImageDataset(zip_path=str(edm_zip), transform=transform)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    batch_imgs, batch_labels = next(iter(loader))
    assert batch_imgs.shape == (4, 3, RES, RES)
    assert batch_labels.shape == (4,)


def test_picklable_for_worker_processes(edm_zip):
    """DataLoader workers pickle the dataset; the zip handle must not break that."""
    transform = dutils.compose_transform(RES, in_channels=3)
    ds = EDMZipImageDataset(zip_path=str(edm_zip), transform=transform)
    _ = ds[0]  # force-open the internal handle
    restored = pickle.loads(pickle.dumps(ds))
    assert restored._zipfile is None  # handle dropped on pickle
    img, _ = restored[1]  # still readable after unpickling (reopens lazily)
    assert img.shape == (3, RES, RES)


def test_empty_zip_raises(tmp_path):
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dataset.json", "{}")
    with pytest.raises(ValueError):
        EDMZipImageDataset(zip_path=str(zip_path))


def test_new_datasets_registered():
    for name in ("ffhq", "imagenet", "afhqv2"):
        assert name in DATASET_REGISTRY
