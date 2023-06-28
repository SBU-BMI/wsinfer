from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import openslide
from PIL import Image

from ..slide_utils import _get_avg_mpp


PathType = Union[str, Path]


def _read_patch_coords(path: PathType) -> np.ndarray:
    """Read HDF5 file of patch coordinates are return numpy array.

    Returned array has shape (num_patches, 4). Each row has values
    [minx, miny, width, height].
    """
    with h5py.File(path, mode="r") as f:
        coords = f["/coords"][()]
        coords_metadata = f["/coords"].attrs
        if "patch_level" not in coords_metadata.keys():
            raise KeyError(
                "Could not find required key 'patch_level' in hdf5 of patch "
                "coordinates. Has the version of CLAM been updated?"
            )
        patch_level = coords_metadata["patch_level"]
        if patch_level != 0:
            raise NotImplementedError(
                f"This script is designed for patch_level=0 but got {patch_level}"
            )
        if coords.ndim != 2:
            raise ValueError(f"expected coords to have 2 dimensions, got {coords.ndim}")
        if coords.shape[1] != 2:
            raise ValueError(
                f"expected second dim of coords to have len 2 but got {coords.shape[1]}"
            )

        if "patch_size" not in coords_metadata.keys():
            raise KeyError("expected key 'patch_size' in attrs of coords dataset")
        # Append width and height values to the coords, so now each row is
        # [minx, miny, width, height]
        wh = np.full_like(coords, coords_metadata["patch_size"])
        coords = np.concatenate((coords, wh), axis=1)

    return coords


class WholeSlideImagePatches(torch.utils.data.Dataset):
    """Dataset of one whole slide image.

    This object retrieves patches from a whole slide image on the fly.

    Parameters
    ----------
    wsi_path : str, Path
        Path to whole slide image file.
    patch_path : str, Path
        Path to npy file with coordinates of input image.
    um_px : float
        Scale of the resulting patches. Use 0.5 for 20x magnification.
    transform : callable, optional
        A callable to modify a retrieved patch. The callable must accept a
        PIL.Image.Image instance and return a torch.Tensor.
    """

    def __init__(
        self,
        wsi_path: PathType,
        patch_path: PathType,
        um_px: float,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ):
        self.wsi_path = wsi_path
        self.patch_path = patch_path
        self.um_px = float(um_px)
        self.transform = transform

        assert Path(wsi_path).exists(), "wsi path not found"
        assert Path(patch_path).exists(), "patch path not found"

        self.oslide = openslide.OpenSlide(self.wsi_path)
        self.slide_mpp = _get_avg_mpp(self.wsi_path)
        self.patches = _read_patch_coords(self.patch_path)

        # The factor by which to resize the patches.
        self.size_factor = self.slide_mpp / self.um_px

        assert self.patches.ndim == 2, "expected 2D array of patch coordinates"
        # x, y, width, height
        assert self.patches.shape[1] == 4, "expected second dimension to have len 4"

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[Image.Image, torch.Tensor], torch.Tensor]:
        coords: Sequence[int] = self.patches[idx]
        assert len(coords) == 4, "expected 4 coords (minx, miny, width, height)"
        minx, miny, width, height = coords

        patch_im = self.oslide.read_region(
            location=(minx, miny), level=0, size=(width, height)
        )
        patch_im = patch_im.convert("RGB")
        # Resize to the expected spacing. We extract the patches at their highest
        # resolution and resize to the prescribed MPP here.
        patch_im = patch_im.resize(
            (self.size_factor * width, self.size_factor * height)
        )

        if self.transform is not None:
            patch_im = self.transform(patch_im)
        if not isinstance(patch_im, (Image.Image, torch.Tensor)):
            raise TypeError(
                f"patch image must be an Image of Tensor, but got {type(patch_im)}"
            )
        return patch_im, torch.as_tensor([minx, miny, width, height])
