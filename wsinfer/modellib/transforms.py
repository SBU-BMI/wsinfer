"""PyTorch image classification transform.

From
https://github.com/pytorch/vision/blob/528651a031a08f9f97cc75bd619a326387708219/torchvision/transforms/_presets.py#L1
"""

from typing import Tuple
from typing import Union

from PIL import Image
import torch
from torchvision.transforms import functional as F

# Get the interpolation mode while catering to older (and newer) versions of
# torchvision and PIL.
if hasattr(F, "InterpolationMode"):
    BICUBIC = F.InterpolationMode.BICUBIC
    BILINEAR = F.InterpolationMode.BILINEAR
    LINEAR = F.InterpolationMode.BILINEAR
    NEAREST = F.InterpolationMode.NEAREST
elif hasattr(Image, "Resampling"):
    BICUBIC = Image.Resampling.BICUBIC
    BILINEAR = Image.Resampling.BILINEAR
    LINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
else:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST
    LINEAR = Image.LINEAR


class PatchClassification(torch.nn.Module):
    def __init__(
        self,
        *,
        resize_size: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        interpolation=BILINEAR,
    ) -> None:
        super().__init__()
        self.resize_size = resize_size
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation

    def __repr__(self):
        return (
            f"PatchClassification(resize_size={self.resize_size}, mean={self.mean},"
            f" std={self.std}, interpolation={self.interpolation})"
        )

    def forward(self, img: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        img = F.resize(
            img, [self.resize_size, self.resize_size], interpolation=self.interpolation
        )
        if not isinstance(img, torch.Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img
