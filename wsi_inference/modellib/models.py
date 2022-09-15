"""Models for whole slide image patch-based classification.

See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7369575/table/tbl3/ for modifications
that were made to the original architectures.
"""

import dataclasses
import pathlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from PIL import Image
import torch
from torch.hub import load_state_dict_from_url
import torchvision

from .inceptionv4 import inceptionv4 as _inceptionv4
from .transforms import PatchClassification

PathType = Union[str, pathlib.Path]


class ModelNotFoundError(Exception):
    ...


class WeightsNotFoundError(Exception):
    ...


@dataclasses.dataclass
class Weights:
    """Container for data associated with a trained model."""

    url: str
    file_name: str
    num_classes: int
    transform: Callable[[Union[Image.Image, torch.Tensor]], torch.Tensor]
    patch_size_pixels: int
    spacing_um_px: float
    class_names: List[str]
    metadata: Dict[str, Any]
    model: Optional[torch.nn.Module] = None

    def __post_init__(self):
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names cannot contain duplicates")
        if len(self.class_names) != self.num_classes:
            raise ValueError("length of class_names must be equal to num_classes")


# Store all available weights for all models.
WEIGHTS: Dict[str, Dict[str, Weights]] = {
    "inceptionv4": {
        "TCGA-BRCA-v1": Weights(
            url="https://stonybrookmedicine.box.com/shared/static/tfwimlf3ygyga1x4fnn03u9y5uio8gqk.pt",  # noqa
            file_name="inceptionv4-brca-20190613-aef40942.pt",
            num_classes=2,
            transform=PatchClassification(
                resize_size=299,
                # TODO: check that the mean and std dev are correct.
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
            patch_size_pixels=350,
            spacing_um_px=88 / 350,
            class_names=["notumor", "tumor"],
            metadata={"patch-size": "350 pixels (88 microns)."},
        ),
    },
    "resnet34": {
        "TCGA-BRCA-v1": Weights(
            url="https://stonybrookmedicine.box.com/shared/static/dv5bxk6d15uhmcegs9lz6q70yrmwx96p.pt",  # noqa
            file_name="resnet34-brca-20190613-01eaf604.pt",
            num_classes=2,
            transform=PatchClassification(
                resize_size=224,
                mean=(0.7238, 0.5716, 0.6779),
                std=(0.1120, 0.1459, 0.1089),
            ),
            patch_size_pixels=350,
            spacing_um_px=88 / 350,
            class_names=["notumor", "tumor"],
            metadata={"patch-size": "350 pixels (88 microns)."},
        ),
        # Original model is on GitHub
        # https://github.com/SBU-BMI/quip_lung_cancer_detection/blob/8eac86e837baa371a98ce2fe08348dcf0400a317/models_cnn/train_lung_john_6classes_netDepth-34_APS-350_randomSeed-2954321_numBenign-80000_0131_1818_bestF1_0.8273143068611924_5.t7
        "TCGA-LUAD-v1": Weights(
            url="https://stonybrookmedicine.box.com/shared/static/d6g9huv1olfu2mt9yaud9xqf9bdqx38i.pt",  # noqa
            file_name="resnet34-luad-20210102-93038ae6.pt",
            num_classes=6,
            transform=PatchClassification(
                resize_size=224,
                mean=(0.8301, 0.6600, 0.8054),
                std=(0.0864, 0.1602, 0.0647),
            ),
            patch_size_pixels=350,
            spacing_um_px=88 / 350,
            # TODO: double check the class names.
            class_names=[
                "NSCLC-Lapidic",
                "NSCLC-Benign",
                "NSCLC-Acinar",
                "NSCLC-Micropapillary",
                "NSCLC-Adeno CA (all)",
                "NSCLC-Solid",
            ],
            metadata={},
        ),
        # Original model is on GitHub
        # https://github.com/SBU-BMI/quip_prad_cancer_detection/tree/d80052e0d098a1211432f9abff086974edd9c669/models_cnn
        # Original file name is
        # RESNET_34_prostate_beatrice_john___1117_1038_0.9533516227597434_87.t7
        #
        # TODO: there seems to be some post-processing... We should double check and
        # implement it. See
        # https://github.com/SBU-BMI/quip_prad_cancer_detection/blob/53870a7db8e48673bee1d60db6f561d39483b859/heatmap_gen_separate_classes/3_thresholded_heatmap_txt.py#L20
        "TCGA-PRAD-v1": Weights(
            url="https://stonybrookmedicine.box.com/shared/static/nxyr5atk2nlvgibck3l0q6rjin2g7n38.pt",  # noqa
            file_name="resnet34-prad-20210101-ea6c004c.pt",
            num_classes=4,
            transform=PatchClassification(
                resize_size=224,
                mean=(0.6462, 0.5070, 0.8055),
                std=(0.1381, 0.1674, 0.1358),
            ),
            # TODO: check these values
            patch_size_pixels=175,
            spacing_um_px=88 / 350,  # TODO: is this correct?
            class_names=["unknown", "grade3", "grade4+5", "benign"],
            metadata={},
        ),
    },
    "vgg16_modified": {
        "TCGA-BRCA-v1": Weights(
            url="https://stonybrookmedicine.box.com/shared/static/197s56yvcrdpan7eu5tq8d4gxvq3xded.pt",  # noqa
            file_name="vgg16-modified-brca-20190613-62bc1b41.pt",
            num_classes=2,
            transform=PatchClassification(
                resize_size=224,
                mean=(0.7238, 0.5716, 0.6779),
                std=(0.1120, 0.1459, 0.1089),
            ),
            patch_size_pixels=350,
            spacing_um_px=88 / 350,
            class_names=["notumor", "tumor"],
            metadata={"patch-size": "350 pixels (88 microns)."},
        )
    },
}


def _get_model_weights(model_name: str, weights: str) -> Weights:
    all_weights_for_model = WEIGHTS.get(model_name)
    if all_weights_for_model is None:
        raise ModelNotFoundError(f"no weights registered for {model_name}")
    weights_obj = all_weights_for_model.get(weights)
    if weights_obj is None:
        raise WeightsNotFoundError(f"'{weights}' weights not found for {model_name}")
    return weights_obj


def _load_state_into_model(model: torch.nn.Module, weights: Weights):
    print("Information about the pretrained weights")
    print("----------------------------------------")
    for k, v in dataclasses.asdict(weights).items():
        if k == "model":  # skip because it's None at this point.
            continue
        print(f"{k} = {v}")
    print("----------------------------------------\n")
    state_dict = load_state_dict_from_url(
        url=weights.url, check_hash=True, file_name=weights.file_name
    )
    model.load_state_dict(state_dict, strict=True)
    return model


def inceptionv4(weights: str = "TCGA-BRCA-v1") -> Weights:
    """Create InceptionV4 model."""
    weights_obj = _get_model_weights("inceptionv4", weights=weights)
    model = _inceptionv4(num_classes=weights_obj.num_classes, pretrained=False)
    model = _load_state_into_model(model=model, weights=weights_obj)
    weights_obj.model = model
    return weights_obj


def resnet34(weights: str = "TCGA-BRCA-v1") -> Weights:
    """Create ResNet34 model."""
    weights_obj = _get_model_weights("resnet34", weights=weights)
    model = torchvision.models.resnet34()
    model.fc = torch.nn.Linear(model.fc.in_features, weights_obj.num_classes)
    model = _load_state_into_model(model=model, weights=weights_obj)
    weights_obj.model = model
    return weights_obj


def vgg16_modified(weights="TCGA-BRCA-v1") -> Weights:
    """Create modified VGG16 model.

    The classifier of this model is
        Linear (25,088, 4096)
        ReLU -> Dropout
        Linear (1024, num_classes)
    """
    weights_obj = _get_model_weights("vgg16_modified", weights=weights)
    model = torchvision.models.vgg16()
    model.classifier = model.classifier[:4]
    in_features = model.classifier[0].in_features
    model.classifier[0] = torch.nn.Linear(in_features, 1024)
    model.classifier[3] = torch.nn.Linear(1024, weights_obj.num_classes)
    model = _load_state_into_model(model=model, weights=weights_obj)
    weights_obj.model = model
    return weights_obj


MODELS = dict(
    inceptionv4=inceptionv4,
    resnet34=resnet34,
    vgg16_modified=vgg16_modified,
)


def list_models() -> List[str]:
    return list(MODELS.keys())


def create_model(model_name: str, weights: str = "TCGA-BRCA-v1") -> Weights:
    """Create a model."""
    if model_name not in MODELS.keys():
        raise ModelNotFoundError(
            f"{model_name} not found. Available models are {MODELS.keys()}"
        )
    model_fn = MODELS[model_name]
    weights_obj = model_fn(weights=weights)
    return weights_obj
