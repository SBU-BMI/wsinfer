"""WSInfer is a toolkit for fast patch-based inference on whole slide images."""

from . import _version
from ._modellib.run_inference import run_inference  # noqa
from ._modellib.run_inference import WholeSlideImagePatches  # noqa

__version__ = _version.get_versions()["version"]

del _version
