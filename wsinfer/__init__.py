"""WSInfer is a toolkit for fast patch-based inference on whole slide images."""

from . import _version
from ._modellib.models import get_model_weights  # noqa
from ._modellib.models import list_all_models_and_weights  # noqa
from ._modellib.models import register_model_weights  # noqa
from ._modellib.run_inference import run_inference  # noqa
from ._modellib.run_inference import WholeSlideImagePatches  # noqa
from ._modellib.transforms import PatchClassification  # noqa

__version__ = _version.get_versions()["version"]

del _version
