"""Download a file from a HuggingFace repo, with a dependency-free fallback.

Prefers ``huggingface_hub.hf_hub_download`` (cached, resumable). If ``huggingface_hub``
is not installed or the download fails for any reason, falls back to a plain ``urllib``
download from the public ``resolve/main`` URL -- mirroring the tutorial notebook, so the
datasets and precomputed PCAs remain reachable without the extra dependency.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)

HF_BASE_URL = "https://huggingface.co"


def hf_resolve_url(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    """Public direct-download URL for a file in a HuggingFace repo."""
    prefix = {"dataset": "datasets/", "space": "spaces/", "model": ""}.get(repo_type, "")
    return f"{HF_BASE_URL}/{prefix}{repo_id}/resolve/main/{filename}"


def hf_download(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = "dataset",
    dest_dir: Optional[Path] = None,
) -> Path:
    """Download ``filename`` from ``repo_id``, returning the local path.

    Tries ``huggingface_hub`` first; on ImportError or any download error, falls back to
    a plain ``urllib`` download from the public resolve URL. When ``dest_dir`` is given
    the file is placed there (and reused on subsequent calls); otherwise huggingface_hub's
    cache is used, with the urllib fallback writing to a local cache directory.
    """
    # Preferred path: huggingface_hub (handles caching, resume, auth).
    try:
        from huggingface_hub import hf_hub_download

        kwargs = {"repo_id": repo_id, "filename": filename, "repo_type": repo_type}
        if dest_dir is not None:
            kwargs["local_dir"] = str(dest_dir)
        return Path(hf_hub_download(**kwargs))
    except Exception as err:  # ImportError or any download failure
        LOGGER.warning(
            "huggingface_hub download of %s failed (%s); falling back to urllib.",
            filename,
            err,
        )

    # Fallback: plain urllib download from the public resolve URL.
    url = hf_resolve_url(repo_id, filename, repo_type)
    if dest_dir is None:
        dest_dir = Path.home() / ".cache" / "local_diffusion_hf"
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(filename).name

    if dest.exists():
        LOGGER.info("Using cached download at %s", dest)
        return dest

    LOGGER.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    return dest
