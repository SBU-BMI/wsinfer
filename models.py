"""Models for whole slide image patch-based classification.

All model functions must have the signature:
    Function[[int, Optional[str]], torch.nn.Module]
"""

import torch
import torchvision


class ModelNotFoundError(Exception):
    ...


def resnet34(num_classes: int, state_dict_path=None) -> torchvision.models.ResNet:
    """Create a ResNet34 with a custom number of output classes."""
    model = torchvision.models.resnet34()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    if state_dict_path is not None:
        print("Loading state dict")
        print(f"  {state_dict_path}")
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    return model


MODELS = dict(resnet34=resnet34)


def create_model(model_name: str, num_classes: int, state_dict_path=None):
    if model_name not in MODELS.keys():
        raise ModelNotFoundError(
            f"{model_name} not found. Available models are {MODELS.keys()}"
        )

    model_fn = MODELS[model_name]
    return model_fn(num_classes=num_classes, state_dict_path=state_dict_path)
