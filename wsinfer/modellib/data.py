from __future__ import annotations

from pathlib import Path
from typing import Callable
from typing import Sequence

import h5py
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from wsinfer.wsi import WSI


def _read_patch_coords(path: str | Path) -> npt.NDArray[np.int_]:
    """Read HDF5 file of patch coordinates are return numpy array.

    Returned array has shape (num_patches, 4). Each row has values
    [minx, miny, width, height].
    """
    with h5py.File(path, mode="r") as f:
        coords: npt.NDArray[np.int_] = f["/coords"][()]
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
    transform : callable, optional
        A callable to modify a retrieved patch. The callable must accept a
        PIL.Image.Image instance and return a torch.Tensor.
    """

    def __init__(
        self,
        wsi_path: str | Path,
        patch_path: str | Path,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ):
        self.wsi_path = wsi_path
        self.patch_path = patch_path
        self.transform = transform

        assert Path(wsi_path).exists(), "wsi path not found"
        assert Path(patch_path).exists(), "patch path not found"

        self.patches = _read_patch_coords(self.patch_path)

        assert self.patches.ndim == 2, "expected 2D array of patch coordinates"
        # x, y, width, height
        assert self.patches.shape[1] == 4, "expected second dimension to have len 4"

    def worker_init(self, worker_id: int | None = None) -> None:
        del worker_id
        self.slide = WSI(self.wsi_path)

    def __len__(self) -> int:
        return self.patches.shape[0]

    def __getitem__(self, idx: int) -> tuple[Image.Image | torch.Tensor, torch.Tensor]:
        coords: Sequence[int] = self.patches[idx]
        assert len(coords) == 4, "expected 4 coords (minx, miny, width, height)"
        minx, miny, width, height = coords

        patch_im = self.slide.read_region(
            location=(minx, miny), level=0, size=(width, height)
        )
        patch_im = patch_im.convert("RGB")

        if self.transform is not None:
            patch_im = self.transform(patch_im)

        return patch_im, torch.as_tensor([minx, miny, width, height])
