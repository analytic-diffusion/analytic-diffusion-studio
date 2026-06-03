"""Dataset registrations backed by EDM-format zip archives on HuggingFace.

These datasets are hosted alongside their precomputed PCAs in the HuggingFace dataset
repo ``binxu/image_datasets_PCAs``. Each ``<name>-<res>x<res>.zip`` archive uses the
NVLabs EDM dataset layout: PNG frames stored *inside* the zip under nested folders
(e.g. ``00000/img00000000.png``) plus a ``dataset.json`` label sidecar. The images are
already at the target resolution and stored as uint8 RGB.

We read frames directly from the zip (no extraction) -- important for large archives
such as ImageNet-64 (~16 GB). Pixels are normalized to the framework's ``[-1, 1]`` range
via ``utils.compose_transform``, matching how the precomputed PCAs were prepared
(``x = 2 * x01 - 1``). The dataset ``name`` + ``resolution`` keys match the
precomputed-PCA keys in ``utils.pca`` (e.g. ``ffhq`` @ 64), so the ``wiener`` and
``pca_locality`` models can reuse the published PCA decomposition.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import List, Optional

from torch.utils.data import Dataset
from PIL import Image

from local_diffusion.configuration import DatasetConfig

from . import utils
from .datasets import DatasetFactoryOutput, register_dataset


LOGGER = logging.getLogger(__name__)


HF_REPO = "binxu/image_datasets_PCAs"

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class EDMZipImageDataset(Dataset):
    """Read images directly from an EDM-format zip archive (no extraction).

    Mirrors the iteration order of NVLabs EDM's ``ImageFolderDataset`` (sorted frame
    names) so the dataset is consistent with the published PCA statistics. Labels in
    ``dataset.json`` are ignored -- a dummy ``0`` is returned to match the rest of the
    framework's ``(image, label)`` convention.
    """

    def __init__(self, zip_path: str, transform=None):
        self._path = str(zip_path)
        self.transform = transform
        # Open once to enumerate frames, then close; per-worker handles are opened lazily.
        with zipfile.ZipFile(self._path) as zf:
            self._image_fnames: List[str] = sorted(
                n for n in zf.namelist() if n.lower().endswith(_IMAGE_EXTS)
            )
        if not self._image_fnames:
            raise ValueError(f"No image frames found in zip archive {self._path}")
        self._zipfile: Optional[zipfile.ZipFile] = None
        LOGGER.info("Found %d frames in %s", len(self._image_fnames), self._path)

    def _zip(self) -> zipfile.ZipFile:
        # Reopen lazily so each DataLoader worker process gets its own handle
        # (zipfile handles are not safe to share across a fork).
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def __getstate__(self):
        # Drop the (unpicklable / fork-unsafe) handle when sent to worker processes.
        return {**self.__dict__, "_zipfile": None}

    def __len__(self) -> int:
        return len(self._image_fnames)

    def __getitem__(self, idx):
        with self._zip().open(self._image_fnames[idx], "r") as f:
            image = Image.open(f).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # dummy label


def _resolve_zip_path(cfg: DatasetConfig, zip_filename: str) -> Path:
    """Return a local path to the dataset zip, downloading from HF if needed."""
    local_path = Path(cfg.root) / zip_filename
    if local_path.exists():
        return local_path

    if not cfg.download:
        raise FileNotFoundError(
            f"Dataset zip not found at {local_path}. Set download=True in config, or "
            f"place {zip_filename} there manually from "
            f"https://huggingface.co/datasets/{HF_REPO}."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "huggingface_hub is required to download HuggingFace image datasets. "
            "Install it with `uv pip install huggingface_hub`."
        ) from exc

    LOGGER.info("Downloading %s from %s...", zip_filename, HF_REPO)
    cached = hf_hub_download(repo_id=HF_REPO, filename=zip_filename, repo_type="dataset")
    return Path(cached)


def _build_edm_zip_dataset(
    cfg: DatasetConfig,
    *,
    zip_filename: str,
    default_resolution: int,
) -> DatasetFactoryOutput:
    resolution = cfg.resolution or default_resolution
    zip_path = _resolve_zip_path(cfg, zip_filename)

    transform = utils.compose_transform(resolution, in_channels=3)
    dataset = EDMZipImageDataset(zip_path=str(zip_path), transform=transform)
    postprocess = utils.get_postprocess_fn()

    return DatasetFactoryOutput(
        dataset=dataset,
        resolution=resolution,
        in_channels=3,
        postprocess=postprocess,
    )


@register_dataset("ffhq")
def build_ffhq(cfg: DatasetConfig) -> DatasetFactoryOutput:
    """FFHQ 64x64 from HuggingFace (``ffhq-64x64.zip``, EDM format)."""
    return _build_edm_zip_dataset(
        cfg, zip_filename="ffhq-64x64.zip", default_resolution=64
    )


@register_dataset("imagenet")
def build_imagenet(cfg: DatasetConfig) -> DatasetFactoryOutput:
    """ImageNet 64x64 from HuggingFace (``imagenet-64x64.zip``, EDM format, ~16 GB)."""
    return _build_edm_zip_dataset(
        cfg, zip_filename="imagenet-64x64.zip", default_resolution=64
    )


@register_dataset("afhqv2")
def build_afhqv2(cfg: DatasetConfig) -> DatasetFactoryOutput:
    """AFHQv2 64x64 from HuggingFace (``afhqv2-64x64.zip``, EDM format).

    Distinct from the ``afhq`` registration, which downloads the full-resolution AFHQv2
    from Dropbox. This variant matches the published 64x64 precomputed PCA.
    """
    return _build_edm_zip_dataset(
        cfg, zip_filename="afhqv2-64x64.zip", default_resolution=64
    )
