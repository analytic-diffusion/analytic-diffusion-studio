# Minimal subset of NVLabs EDM's dnnlib.util, vendored for checkpoint loading.
#
# Original: https://github.com/NVlabs/edm (dnnlib/util.py)
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0): http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Only ``EasyDict`` is required to unpickle EDM checkpoints (it is the sole
# ``dnnlib`` symbol referenced by torch_utils.persistence / torch_utils.misc and by
# the persistent class sources embedded in the pickles). The rest of the upstream
# module (network download, logging, etc.) is intentionally omitted.

from __future__ import annotations


class EasyDict(dict):
    """Dictionary with attribute-style access (``d.key`` == ``d['key']``)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]
