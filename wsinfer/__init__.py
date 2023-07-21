"""WSInfer is a toolkit for fast patch-based inference on whole slide images."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from .modellib.run_inference import WholeSlideImagePatches  # noqa
from .modellib.run_inference import run_inference  # noqa

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.unknown"


# Patch Zarr. See:
# https://github.com/bayer-science-for-a-better-life/tiffslide/issues/72#issuecomment-1627918238
# https://github.com/zarr-developers/zarr-python/pull/1454
def _patch_zarr_kvstore():
    from zarr.storage import KVStore

    def _zarr_KVStore___contains__(self, key):
        return key in self._mutable_mapping

    KVStore.__contains__ = _zarr_KVStore___contains__


_patch_zarr_kvstore()
