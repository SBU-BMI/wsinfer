import dataclasses
import warnings
from typing import Callable, Dict, Union

import safetensors.torch
import timm
import torch
import wsinfer_zoo
from wsinfer_zoo.client import HFModel, HFModelTorchScript, HFModelWeightsOnly, Model

# Imported for side effects of registering model.
from ..errors import UnknownArchitectureError
from .custom_models import inceptionv4_no_batchnorm as _  # noqa
from .custom_models.resnet_preact import resnet34_preact as _resnet34_preact
from .custom_models.vgg16mod import vgg16mod as _vgg16mod


@dataclasses.dataclass
class LocalModel(Model):
    ...


@dataclasses.dataclass
class LocalModelTorchScript(Model):
    ...


@dataclasses.dataclass
class LocalModelWeightsOnly(Model):
    ...


# Container for all architectures we can use that are not in timm.
# The values are functions, which are expected to accept one integer
# argument for the number of classes the model outputs.
_architecture_registry: Dict[str, Callable[[int], torch.nn.Module]] = {
    "preactresnet34": _resnet34_preact,
    "vgg16mod": _vgg16mod,
}


def _create_model(name: str, num_classes: int) -> torch.nn.Module:
    """Return a torch model architecture."""
    if name in _architecture_registry.keys():
        return _architecture_registry[name](num_classes)
    else:
        if name not in timm.list_models():
            raise UnknownArchitectureError(f"unknown architecture: '{name}'")
        return timm.create_model(name, num_classes=num_classes)


def get_registered_model(
    name: str, torchscript: bool = False, safetensors: bool = False
) -> HFModel:
    model = wsinfer_zoo.registry.get_model_by_name(name=name)
    if torchscript:
        return model.load_model_torchscript()
    else:
        return model.load_model_weights(safetensors=safetensors)


def get_pretrained_torch_module(model: Union[LocalModel, HFModel]) -> torch.nn.Module:
    """Get a PyTorch Module with weights loaded."""

    if isinstance(model, (HFModelTorchScript, LocalModelTorchScript)):
        return torch.jit.load(model.model_path, map_location="cpu")

    elif isinstance(model, (HFModelWeightsOnly, LocalModelWeightsOnly)):
        arch = _create_model(
            name=model.config.architecture, num_classes=model.config.num_classes
        )
        # FIXME: this might cause problems for us down the line. Is it this
        # specific enough?
        if model.model_path.endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(model.model_path)
        else:
            state_dict = torch.load(model.model_path, map_location="cpu")
        arch.load_state_dict(state_dict)
        return arch
    else:
        raise ValueError(
            f"expected Model or ModelWeightsOnly instance but got {type(model)}"
        )


def jit_compile(
    model: torch.nn.Module,
) -> Union[torch.jit.ScriptModule, torch.nn.Module, Callable]:
    """JIT-compile a model for inference.

    A torchscript model may be JIT compiled here as well.
    """
    noncompiled = model
    device = next(model.parameters()).device
    # Attempt to script. If it fails, return the original.
    test_input = torch.ones(1, 3, 224, 224).to(device)
    w = "Warning: could not JIT compile the model. Using non-compiled model instead."

    # PyTorch 2.x has torch.compile but it does not work when applied
    # to TorchScript models.
    if hasattr(torch, "compile") and not isinstance(model, torch.jit.ScriptModule):
        # Try to get the most optimized model.
        try:
            return torch.compile(model, fullgraph=True, mode="max-autotune")
        except Exception:
            pass
        try:
            return torch.compile(model, mode="max-autotune")
        except Exception:
            pass
        try:
            return torch.compile(model)
        except Exception:
            warnings.warn(w)
            return noncompiled
    # For pytorch 1.x, use torch.jit.script.
    else:
        try:
            mjit = torch.jit.script(model)
            with torch.no_grad():
                mjit(test_input)
        except Exception:
            warnings.warn(w)
            return noncompiled
        # Now that we have scripted the model, try to optimize it further. If that
        # fails, return the scripted model.
        try:
            mjit_frozen = torch.jit.freeze(mjit)
            mjit_opt = torch.jit.optimize_for_inference(mjit_frozen)
            with torch.no_grad():
                mjit_opt(test_input)
            return mjit_opt
        except Exception:
            return mjit
