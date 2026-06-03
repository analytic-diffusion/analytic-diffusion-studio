"""Tests for the HuggingFace download helper and its urllib fallback (no network)."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pytest

from local_diffusion.utils.hf_download import hf_download, hf_resolve_url


def test_resolve_url_dataset_and_model():
    assert (
        hf_resolve_url("binxu/image_datasets_PCAs", "cifar32_PCA.pt", "dataset")
        == "https://huggingface.co/datasets/binxu/image_datasets_PCAs/resolve/main/cifar32_PCA.pt"
    )
    assert (
        hf_resolve_url("some/model", "weights.pt", "model")
        == "https://huggingface.co/some/model/resolve/main/weights.pt"
    )


def test_urllib_fallback_when_hub_unavailable(tmp_path, monkeypatch):
    # Setting the module to None makes `from huggingface_hub import ...` raise ImportError.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    captured = {}

    def fake_urlretrieve(url, dest):
        captured["url"] = url
        Path(dest).write_bytes(b"payload")
        return dest, None

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    out = hf_download(
        "binxu/image_datasets_PCAs", "ffhq64_PCA.pt",
        repo_type="dataset", dest_dir=tmp_path,
    )

    assert out == tmp_path / "ffhq64_PCA.pt"
    assert out.read_bytes() == b"payload"
    assert captured["url"].endswith(
        "/datasets/binxu/image_datasets_PCAs/resolve/main/ffhq64_PCA.pt"
    )


def test_fallback_reuses_existing_file(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    existing = tmp_path / "ffhq64_PCA.pt"
    existing.write_bytes(b"cached")

    def boom(url, dest):
        raise AssertionError("must not re-download when the file already exists")

    monkeypatch.setattr(urllib.request, "urlretrieve", boom)

    out = hf_download("binxu/image_datasets_PCAs", "ffhq64_PCA.pt", dest_dir=tmp_path)
    assert out == existing
    assert out.read_bytes() == b"cached"
