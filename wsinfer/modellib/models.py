import dataclasses
import hashlib
import os
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from PIL import Image
import timm
import torch
from torch.hub import load_state_dict_from_url
import yaml

from .inceptionv4 import inceptionv4 as _inceptionv4
from .inceptionv4_no_batchnorm import inceptionv4 as _inceptionv4_no_bn
from .resnet_preact import resnet34_preact as _resnet34_preact
from .vgg16mod import vgg16mod as _vgg16mod
from .transforms import PatchClassification


class WsinferException(Exception):
    "Base class for wsinfer exceptions."


class UnknownArchitectureError(WsinferException):
    """Architecture is unknown and cannot be found."""


class ModelWeightsNotFound(WsinferException):
    """Model weights are not found, likely because they are not in the registry."""


class DuplicateModelWeights(WsinferException):
    """A duplicate key was passed to the model weights registry."""


PathType = Union[str, Path]


def _sha256sum(path: PathType) -> str:
    """Calculate SHA256 of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(1024 * 64)  # 64 kb
            if not data:
                break
            sha.update(data)
    return sha.hexdigest()


@dataclasses.dataclass
class Weights:
    """Container for data associated with a trained model."""

    name: str
    architecture: str
    num_classes: int
    transform: Callable[[Union[Image.Image, torch.Tensor]], torch.Tensor]
    patch_size_pixels: int
    spacing_um_px: float
    class_names: List[str]
    url: Optional[str] = None
    url_file_name: Optional[str] = None
    file: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names cannot contain duplicates")
        if len(self.class_names) != self.num_classes:
            raise ValueError("length of class_names must be equal to num_classes")

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            d = yaml.safe_load(f)

        if not isinstance(d, dict):
            raise ValueError("expected YAML config to be a dictionary")

        # Validate contents.
        # Validate keys.
        required_keys = [
            "name",
            "architecture",
            "num_classes",
            "transform",
            "patch_size_pixels",
            "spacing_um_px",
            "class_names",
        ]
        optional_keys = ["url", "url_file_name", "file", "metadata"]
        all_keys = required_keys + optional_keys
        for req_key in required_keys:
            if req_key not in d.keys():
                raise KeyError(f"required key not found: '{req_key}'")
        unknown_keys = [k for k in d.keys() if k not in all_keys]
        if unknown_keys:
            raise KeyError(f"unknown keys: {unknown_keys}")
        for req_key in ["resize_size", "mean", "std"]:
            if req_key not in d["transform"].keys():
                raise KeyError(
                    f"required key not found in 'transform' section: '{req_key}'"
                )

        # Either 'url' or 'file' is required. If 'url' is used, then 'url_file_name' is
        # required.
        if "url" not in d.keys() and "file" not in d.keys():
            raise KeyError("'url' or 'file' must be provided")
        if "url" in d.keys() and "file" in d.keys():
            raise KeyError("only on of 'url' and 'file' can be used")
        if "url" in d.keys() and "url_file_name" not in d.keys():
            raise KeyError("when using 'url', 'url_file_name' must also be provided")

        # Validate types.
        if not isinstance("architecture", str):
            raise ValueError("'architecture' must be a string")
        if not isinstance("name", str):
            raise ValueError("'name' must be a string")
        if "url" in d.keys() and not isinstance(d["url"], str):
            raise ValueError("'url' must be a string")
        if "url_file_name" in d.keys() and not isinstance(d["url"], str):
            raise ValueError("'url_file_name' must be a string")
        if not isinstance(d["num_classes"], int):
            raise ValueError("'num_classes' must be an integer")
        if not isinstance(d["transform"]["resize_size"], int):
            raise ValueError("'transform.resize_size' must be an integer")
        if not isinstance(d["transform"]["mean"], list):
            raise ValueError("'transform.mean' must be a list")
        if not all(isinstance(num, float) for num in d["transform"]["mean"]):
            raise ValueError("'transform.mean' must be a list of floats")
        if not isinstance(d["transform"]["std"], list):
            raise ValueError("'transform.std' must be a list")
        if not all(isinstance(num, float) for num in d["transform"]["std"]):
            raise ValueError("'transform.std' must be a list of floats")
        if not isinstance(d["patch_size_pixels"], int) or d["patch_size_pixels"] <= 0:
            raise ValueError("patch_size_pixels must be a positive integer")
        if not isinstance(d["spacing_um_px"], float) or d["spacing_um_px"] <= 0:
            raise ValueError("spacing_um_px must be a positive float")
        if not isinstance(d["class_names"], list):
            raise ValueError("'class_names' must be a list")
        if not all(isinstance(c, str) for c in d["class_names"]):
            raise ValueError("'class_names' must be a list of strings")

        # Validate values.
        if len(d["transform"]["mean"]) != 3:
            raise ValueError("transform.mean must be a list of three numbers")
        if len(d["transform"]["std"]) != 3:
            raise ValueError("transform.std must be a list of three numbers")
        if len(d["class_names"]) != len(set(d["class_names"])):
            raise ValueError("duplicate values found in 'class_names'")
        if len(d["class_names"]) != d["num_classes"]:
            raise ValueError("mismatch between length of class_names and num_classes.")
        if "file" in d.keys():
            file = Path(path).parent / d["file"]
            file = file.resolve()
            if not file.exists():
                raise FileNotFoundError(f"'file' not found: {file}")

        transform = PatchClassification(
            resize_size=d["transform"]["resize_size"],
            mean=d["transform"]["mean"],
            std=d["transform"]["std"],
        )
        return Weights(
            name=d["name"],
            architecture=d["architecture"],
            url=d.get("url"),
            url_file_name=d.get("url_file_name"),
            file=d.get("file"),
            num_classes=d["num_classes"],
            transform=transform,
            patch_size_pixels=d["patch_size_pixels"],
            spacing_um_px=d["spacing_um_px"],
            class_names=d["class_names"],
        )

    def load_model(self):
        model = _create_model(name=self.architecture, num_classes=self.num_classes)

        # Load state dict.
        if self.url and self.url_file_name:
            state_dict = load_state_dict_from_url(
                url=self.url, check_hash=True, file_name=self.url_file_name
            )
        elif self.file:
            state_dict = torch.load(self.file, map_location="cpu")
        else:
            raise RuntimeError("cannot find weights")

        model.load_state_dict(state_dict, strict=True)
        return model

    def get_sha256_of_weights(self) -> str:
        if self.url and self.url_file_name:
            p = Path(torch.hub.get_dir()) / "checkpoints" / self.url_file_name
        elif self.file:
            p = Path(self.file)
        else:
            raise RuntimeError("cannot find path to weights")
        sha = _sha256sum(p)
        return sha


# Container for all models we can use that are not in timm.
_model_registry: Dict[str, Callable[[int], torch.nn.Module]] = {
    "inceptionv4": _inceptionv4,
    "inceptionv4nobn": _inceptionv4_no_bn,
    "preactresnet34": _resnet34_preact,
    "vgg16mod": _vgg16mod,
}


def _create_model(name: str, num_classes: int) -> torch.nn.Module:
    """Return a torch model architecture."""
    if name in _model_registry.keys():
        return _model_registry[name](num_classes)
    else:
        if name not in timm.list_models():
            raise UnknownArchitectureError(f"unknown architecture: '{name}'")
        return timm.create_model(name, num_classes=num_classes)


# Keys are tuple of (architecture, weights_name).
_known_model_weights: Dict[Tuple[str, str], Weights] = {}


def register_model_weights(root: Path):
    modeldefs = list(root.glob("*.yml")) + list(root.glob("*.yaml"))
    for modeldef in modeldefs:
        w = Weights.from_yaml(modeldef)
        if w.architecture not in timm.list_models() + list(_model_registry.keys()):
            raise UnknownArchitectureError(f"{w.architecture} implementation not found")
        key = (w.architecture, w.name)
        if key in _known_model_weights:
            raise DuplicateModelWeights("")
        _known_model_weights[key] = w


def get_model_weights(architecture: str, name: str) -> Weights:
    """Get weights object for an architecture and weights name."""
    key = (architecture, name)
    try:
        return _known_model_weights[key]
    except KeyError:
        raise ModelWeightsNotFound(
            f"model weights are not found for architecture '{architecture}' and "
            f"weights name '{name}'. Available models are"
            f"{list_all_models_and_weights()}."
        )


register_model_weights(Path(__file__).parent / ".." / "modeldefs")

# Register any user-supplied configurations.
wsinfer_path = os.environ.get("WSINFER_PATH")
if wsinfer_path is not None:
    for path in wsinfer_path.split(":"):
        register_model_weights(Path(path))
    del path
del wsinfer_path


def list_all_models_and_weights() -> List[Tuple[str, str]]:
    """Return list of tuples of `(model_name, weights_name)` with available pairs."""
    vals = list(_known_model_weights.keys())
    vals.sort()
    return vals
