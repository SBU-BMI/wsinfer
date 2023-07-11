"""PyTorch image classification transform."""

from __future__ import annotations

from typing import List

from torchvision import transforms
from wsinfer_zoo.client import TransformConfigurationItem

# The subset of transforms known to the wsinfer config spec.
# This can be expanded in the future as needs arise.
_name_to_tv_cls = {
    "Resize": transforms.Resize,
    "ToTensor": transforms.ToTensor,
    "Normalize": transforms.Normalize,
}


def make_compose_from_transform_config(
    list_of_transforms: List[TransformConfigurationItem],
) -> transforms.Compose:
    """Create a torchvision Compose instance from configuration of transforms."""
    all_t: List = []
    for t in list_of_transforms:
        cls = _name_to_tv_cls[t.name]
        kwargs = t.arguments or {}
        all_t.append(cls(**kwargs))
    return transforms.Compose(all_t)
