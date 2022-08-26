"""Models for whole slide image patch-based classification.

See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7369575/table/tbl3/ for modifications
that were made to the original architectures.

All model functions must have the signature:
    Function[[int, Optional[str]], torch.nn.Module]
"""

import pathlib
import typing

import torch
from torch.hub import load_state_dict_from_url
import torchvision

from .inceptionv4 import inceptionv4 as _inceptionv4
from .inceptionv4 import InceptionV4 as _InceptionV4

# TODO: consider adding the color normalization and input size here. Or maybe as a
# command line argument.
# One way forward is to follow in Torchvision's footsteps and create a Weights dataclass
# to hold the url, transform function, and metadata. That is easy enough, but then how
# do we associate each model with its available weights? Torchvision uses subclassed
# enums but for this project, let's go with something simpler.

PathType = typing.Union[str, pathlib.Path]


def inceptionv4(num_classes: int, weights="TCGA-BRCA-v1") -> _InceptionV4:
    """Create InceptionV4 model."""
    model = _inceptionv4(num_classes=num_classes, pretrained=False)
    if weights == "TCGA-BRCA-v1":
        print("Loading state dict")
        state_dict = load_state_dict_from_url(
            url="https://stonybrookmedicine.box.com/shared/static/tfwimlf3ygyga1x4fnn03u9y5uio8gqk.pt",  # noqa
            check_hash=True,
            file_name="inceptionv4-brca-20190613-aef40942.pt",
        )
        model.load_state_dict(state_dict, strict=True)
    else:
        raise NotImplementedError("only TCGA-BRCA-v1 is available now.")
    return model


def resnet34(num_classes: int, weights="TCGA-BRCA-v1") -> torchvision.models.ResNet:
    """Create ResNet34 model."""
    model = torchvision.models.resnet34()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    if weights == "TCGA-BRCA-v1":
        print("Loading state dict")
        state_dict = load_state_dict_from_url(
            url="https://stonybrookmedicine.box.com/shared/static/dv5bxk6d15uhmcegs9lz6q70yrmwx96p.pt",  # noqa
            check_hash=True,
            file_name="resnet34-brca-20190613-01eaf604.pt",
        )
        model.load_state_dict(state_dict, strict=True)
    else:
        raise NotImplementedError("only TCGA-BRCA-v1 is available now.")
    return model


def vgg16_modified(num_classes: int, weights="TCGA-BRCA-v1") -> torch.nn.Module:
    """Create modified VGG16 model.

    The classifier of this model is
        Linear (25,088, 4096)
        ReLU -> Dropout
        Linear (1024, num_classes)
    """
    model = torchvision.models.vgg16()
    model.classifier = model.classifier[:4]
    in_features = model.classifier[0].in_features
    model.classifier[0] = torch.nn.Linear(in_features, 1024)
    model.classifier[3] = torch.nn.Linear(1024, num_classes)
    if weights == "TCGA-BRCA-v1":
        print("Loading state dict")
        state_dict = load_state_dict_from_url(
            url="https://stonybrookmedicine.box.com/shared/static/197s56yvcrdpan7eu5tq8d4gxvq3xded.pt",  # noqa
            check_hash=True,
            file_name="vgg16-modified-brca-20190613-62bc1b41.pt",
        )
        model.load_state_dict(state_dict, strict=True)
    else:
        raise NotImplementedError("only TCGA-BRCA-v1 is available now.")
    return model


MODELS: typing.Dict[str, typing.Callable[..., torch.nn.Module]] = dict(
    inceptionv4=inceptionv4,
    resnet34=resnet34,
    vgg16_modified=vgg16_modified,
)


class ModelNotFoundError(Exception):
    ...


def list_models() -> typing.List[str]:
    return list(MODELS.keys())


def create_model(
    model_name: str,
    num_classes: int,
    weights: str = "TCGA-BRCA-v1",
) -> torch.nn.Module:
    """Create a model."""
    if model_name not in MODELS.keys():
        raise ModelNotFoundError(
            f"{model_name} not found. Available models are {MODELS.keys()}"
        )
    model_fn = MODELS[model_name]
    return model_fn(num_classes=num_classes, weights=weights)
