"""Models for whole slide image patch-based classification.

See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7369575/table/tbl3/ for modifications
that were made to the original architectures.

All model functions must have the signature:
    Function[[int, Optional[str]], torch.nn.Module]
"""

import torch
import torchvision

# TODO: consider adding the color normalization and input size here. Or maybe as a
# command line argument.


def inceptionv4(num_classes: int, state_dict_path=None) -> torch.nn.Module:
    """Create InceptionV4 model."""
    import pretrainedmodels

    model = pretrainedmodels.inceptionv4(num_classes=num_classes, pretrained=False)
    if state_dict_path is not None:
        print("Loading state dict")
        print(f"  {state_dict_path}")
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    return model


def resnet34(num_classes: int, state_dict_path=None) -> torchvision.models.ResNet:
    """Create ResNet34 model."""
    model = torchvision.models.resnet34()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    if state_dict_path is not None:
        print("Loading state dict")
        print(f"  {state_dict_path}")
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    return model


def vgg16_modified(num_classes: int, state_dict_path=None) -> torch.nn.Module:
    """Create modified VGG16 model.

    The classifier of this model is
        Linear (25,088, 4096)
        ReLU → Dropout
        Linear (1024, num_classes)
    """
    model = torchvision.models.vgg16()
    model.classifier = model.classifier[:4]
    in_features = model.classifier[0].in_features
    model.classifier[0] = torch.nn.Linear(in_features, 1024)
    model.classifier[3] = torch.nn.Linear(1024, num_classes)
    if state_dict_path is not None:
        print("Loading state dict")
        print(f"  {state_dict_path}")
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    return model


MODELS = dict(inceptionv4=inceptionv4, resnet34=resnet34, vgg16_modified=vgg16_modified)


class ModelNotFoundError(Exception):
    ...


def create_model(model_name: str, num_classes: int, state_dict_path=None):
    """Create a model."""
    if model_name not in MODELS.keys():
        raise ModelNotFoundError(
            f"{model_name} not found. Available models are {MODELS.keys()}"
        )
    model_fn = MODELS[model_name]
    return model_fn(num_classes=num_classes, state_dict_path=state_dict_path)
