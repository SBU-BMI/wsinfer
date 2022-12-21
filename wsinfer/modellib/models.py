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
from typing import Tuple
from typing import Union

from PIL import Image
import torch
from torch.hub import load_state_dict_from_url
import torchvision

from .inceptionv4 import inceptionv4 as _inceptionv4
from .inceptionv4_no_batchnorm import inceptionv4 as _inceptionv4_no_bn
from .resnet_preact import resnet34_preact as _resnet34_preact
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
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["notumor", "tumor"],
            metadata={"patch-size": "350 pixels (87.5 microns)"},
        ),
        # This uses an implementation without batchnorm. Model was trained with TF Slim
        # and weights were converted to PyTorch (see 'scripts' directory).
        "TCGA-TILs-v1": Weights(
            url="https://stonybrookmedicine.box.com/shared/static/sz1gpc6u3mftadh4g6x3csxnpmztj8po.pt",  # noqa
            file_name="inceptionv4-tils-v1-20200920-e3e72cd2.pt",
            num_classes=2,
            transform=PatchClassification(
                resize_size=299, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),
            patch_size_pixels=100,
            spacing_um_px=0.5,
            class_names=["notils", "tils"],
            metadata={
                "publication": "https://doi.org/10.3389/fonc.2021.806603",
                "notes": (
                    "Implementation does not use batchnorm. Original model was trained"
                    " with TF Slim and converted to PyTorch format."
                ),
            },
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
            spacing_um_px=0.25,
            class_names=["notumor", "tumor"],
            metadata={"patch-size": "350 pixels (87.5 microns)."},
        ),
        # Original model is on GitHub
        # https://github.com/SBU-BMI/quip_lung_cancer_detection/blob/8eac86e837baa371a98ce2fe08348dcf0400a317/models_cnn/train_lung_john_6classes_netDepth-34_APS-350_randomSeed-2954321_numBenign-80000_0131_1818_bestF1_0.8273143068611924_5.t7
        "TCGA-LUAD-v1": Weights(
            url="https://stonybrookmedicine.box.com/shared/static/d6g9huv1olfu2mt9yaud9xqf9bdqx38i.pt",  # noqa
            file_name="resnet34-luad-20210102-93038ae6.pt",
            num_classes=6,
            transform=PatchClassification(
                resize_size=224,
                # Mean and std from
                # https://github.com/SBU-BMI/quip_lung_cancer_detection/blob/8eac86e837baa371a98ce2fe08348dcf0400a317/prediction_6classes/tumor_pred/pred.py#L29-L30
                mean=(0.8301, 0.6600, 0.8054),
                std=(0.0864, 0.1602, 0.0647),
            ),
            patch_size_pixels=350,
            spacing_um_px=0.5,
            class_names=[
                "lepidic",
                "benign",
                "acinar",
                "micropapillary",
                "mucinous",
                "solid",
            ],
            metadata={},
        ),
        # Original model is on GitHub
        # https://github.com/SBU-BMI/quip_prad_cancer_detection/tree/d80052e0d098a1211432f9abff086974edd9c669/models_cnn
        # Original file name is
        # RESNET_34_prostate_beatrice_john___1117_1038_0.9533516227597434_87.t7
        "TCGA-PRAD-v1": Weights(
            url="https://stonybrookmedicine.box.com/shared/static/nxyr5atk2nlvgibck3l0q6rjin2g7n38.pt",  # noqa
            file_name="resnet34-prad-20210101-ea6c004c.pt",
            num_classes=3,
            transform=PatchClassification(
                resize_size=224,
                # Mean and std from
                # https://github.com/SBU-BMI/quip_prad_cancer_detection/blob/b71d8440eab090cb789281b33fbf89011e924fb9/prediction_3classes/tumor_pred/pred.py#L27-L28
                mean=(0.6462, 0.5070, 0.8055),
                std=(0.1381, 0.1674, 0.1358),
            ),
            patch_size_pixels=175,
            spacing_um_px=0.5,
            class_names=["grade3", "grade4+5", "benign"],
            metadata={},
        ),
    },
    "resnet34_preact": {
        "TCGA-PAAD-v1": Weights(
            url="https://stonybrookmedicine.box.com/shared/static/sol1h9aqrh8lynzc6kidw1lsoeks20hh.pt",  # noqa
            file_name="preactresnet34-paad-20210101-7892b41f.pt",
            num_classes=1,
            transform=PatchClassification(
                resize_size=224,
                mean=(0.7238, 0.5716, 0.6779),
                std=(0.1120, 0.1459, 0.1089),
            ),
            patch_size_pixels=350,
            # Patches are 525.1106 microns.
            # Patch of 2078 pixels @ 0.2527 mpp is 350 pixels at our target spacing.
            # (2078 * 0.2527) / 350
            spacing_um_px=1.500316,
            class_names=["tumor"],
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
            spacing_um_px=0.25,
            class_names=["notumor", "tumor"],
            metadata={"patch-size": "350 pixels (87.5 microns)"},
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


def inceptionv4(weights: str) -> Weights:
    """Create InceptionV4 model."""
    weights_obj = _get_model_weights("inceptionv4", weights=weights)
    if weights == "TCGA-TILs-v1":
        # TCGA-TILs-v1 model uses inceptionv4 without batchnorm.
        model = _inceptionv4_no_bn(weights_obj.num_classes, pretrained=False)
    else:
        model = _inceptionv4(num_classes=weights_obj.num_classes, pretrained=False)
    model = _load_state_into_model(model=model, weights=weights_obj)
    weights_obj.model = model
    return weights_obj


def resnet34(weights: str) -> Weights:
    """Create ResNet34 model."""
    weights_obj = _get_model_weights("resnet34", weights=weights)
    model = torchvision.models.resnet34()
    model.fc = torch.nn.Linear(model.fc.in_features, weights_obj.num_classes)
    model = _load_state_into_model(model=model, weights=weights_obj)
    weights_obj.model = model
    return weights_obj


def resnet34_preact(weights: str) -> Weights:
    """Create ResNet34-Preact model."""
    weights_obj = _get_model_weights("resnet34_preact", weights=weights)
    model = _resnet34_preact()
    model.linear = torch.nn.Linear(model.linear.in_features, weights_obj.num_classes)
    model = _load_state_into_model(model=model, weights=weights_obj)
    weights_obj.model = model
    return weights_obj


def vgg16_modified(weights: str) -> Weights:
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


MODEL_NAME_TO_FUNC = dict(
    inceptionv4=inceptionv4,
    resnet34=resnet34,
    resnet34_preact=resnet34_preact,
    vgg16_modified=vgg16_modified,
)
if not set(WEIGHTS.keys()).issubset(MODEL_NAME_TO_FUNC.keys()):
    raise RuntimeError("Not all model functions are defined.")


def list_models() -> List[str]:
    return list(WEIGHTS.keys())


def list_available_weights_for_model(model_name: str) -> List[str]:
    try:
        return list(WEIGHTS[model_name].keys())
    except KeyError:
        return []


def list_available_models_for_weights(weights: str) -> List[str]:
    res = []
    for model_name, weights_dict in WEIGHTS.items():
        for weights_name in weights_dict.keys():
            if weights_name == weights:
                res.append(model_name)
    return res


def list_all_models_and_weights() -> List[Tuple[str, str]]:
    """Return list of tuples of `(model_name, weights_name)` with available pairs."""
    res = []
    for m in WEIGHTS.keys():
        res.extend([(m, w) for w in WEIGHTS[m].keys()])
    return res


def create_model(model_name: str, weights: str) -> Weights:
    """Create a model."""
    if model_name not in list_models():
        raise ModelNotFoundError(
            f"'{model_name}' not found. Available models are {list_models()}"
        )
    if weights not in list_available_weights_for_model(model_name):
        raise WeightsNotFoundError(
            f"The weights available for '{model_name}' are"
            f" {list_available_weights_for_model(model_name)}.\n"
            f"The weights '{weights}' are available for the following models:"
            f" {list_available_models_for_weights(weights)}."
        )

    # TODO: figure out how to pair model_fn with the Weights instances.
    model_fn = MODEL_NAME_TO_FUNC[model_name]
    weights_obj = model_fn(weights=weights)
    return weights_obj
