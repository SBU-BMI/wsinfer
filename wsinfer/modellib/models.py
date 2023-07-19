from __future__ import annotations

import dataclasses
import warnings
from typing import Callable
from typing import Union

import torch
import wsinfer_zoo
from wsinfer_zoo.client import HFModelTorchScript
from wsinfer_zoo.client import Model


@dataclasses.dataclass
class LocalModelTorchScript(Model):
    ...


def get_registered_model(name: str) -> HFModelTorchScript:
    registry = wsinfer_zoo.client.load_registry()
    model = registry.get_model_by_name(name=name)
    return model.load_model_torchscript()


def get_pretrained_torch_module(
    model: HFModelTorchScript | LocalModelTorchScript,
) -> torch.nn.Module:
    """Get a PyTorch Module with weights loaded."""
    return torch.jit.load(model.model_path, map_location="cpu")


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
