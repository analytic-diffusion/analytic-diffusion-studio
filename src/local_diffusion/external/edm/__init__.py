"""Vendored minimal subset of NVLabs EDM, for loading EDM UNet checkpoints.

Upstream: https://github.com/NVlabs/edm
Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.
Licensed under CC BY-NC-SA 4.0 (see ``LICENSE.txt`` in this directory). Only the files
needed to (un)pickle and instantiate the released ``EDMPrecond`` / ``SongUNet`` networks
are included: ``torch_utils/persistence.py``, ``torch_utils/misc.py``,
``training/networks.py`` and a one-symbol ``dnnlib`` shim.

EDM checkpoints are pickled ``nn.Module`` objects whose pickle stores **top-level**
module paths (``torch_utils.persistence``, ``dnnlib``). ``ensure_importable()`` puts this
directory on ``sys.path`` so those names resolve to the vendored copies -- no clone of the
upstream repo required.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_EDM_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_importable() -> None:
    """Make the vendored ``torch_utils`` / ``dnnlib`` importable as top-level modules."""
    if _EDM_DIR not in sys.path:
        sys.path.insert(0, _EDM_DIR)


def load_edm_pickle(path: str | Path, key: Optional[str] = "ema") -> Any:
    """Unpickle an official EDM ``.pkl`` and return ``data[key]`` (the EMA network).

    Pass ``key=None`` to return the whole snapshot dict.
    """
    ensure_importable()
    import torch  # noqa: F401  (embedded persistent-class sources import torch)

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data[key] if key is not None else data


def build_edm_network(init_kwargs: Dict[str, Any]):
    """Instantiate a fresh ``EDMPrecond`` from saved ``init_kwargs`` (for converted ckpts)."""
    ensure_importable()  # networks.py does `from torch_utils import persistence`
    from .training import networks

    return networks.EDMPrecond(**init_kwargs)
