from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
from PIL import Image

from ..wsi import WSI
from ..wsi import get_avg_mpp
from .patch import get_multipolygon_from_binary_arr
from .patch import get_nonoverlapping_patch_coordinates_within_polygon
from .segment import segment_tissue

logger = logging.getLogger(__name__)


def segment_and_patch(
    path: str | Path,
    patch_size_px: int,
    patch_spacing_um_px: float,
    thumbsize: tuple[int, int] = (2048, 2048),
    median_filter_size: int = 7,
    closing_kernel_size: int = 6,
    min_hole_size: int = 64,
    min_object_size: int = 64,
):
    logger.info(f"Segmenting and patching slide {path}")
    slide = WSI(path)

    mpp = get_avg_mpp(path)
    logger.info(f"MPP={mpp} micrometers per pixel")

    logger.info(
        f"Requested patch size of {patch_size_px} px @ {patch_spacing_um_px} um/px"
    )
    logger.info(
        f"Scaling patch size by {patch_spacing_um_px / mpp}"
        f" | patch_spacing_um_px / mpp ({patch_spacing_um_px} / {mpp})"
    )
    patch_size = round(patch_size_px * patch_spacing_um_px / mpp)
    logger.info(f"Final patch size is {patch_size}")

    # Segment tissue into a binary mask.
    if len(thumbsize) != 2:
        raise ValueError(f"Length of 'thumbsize' must be 2 but got {len(thumbsize)}")
    thumb: Image.Image = slide.get_thumbnail(thumbsize)
    # TODO: allow the min hole size and min object size to be set in physical units.
    arr = segment_tissue(
        thumb,
        median_filter_size=median_filter_size,
        closing_kernel_size=closing_kernel_size,
        min_hole_size=min_hole_size,
        min_object_size=min_object_size,
    )
    if not np.issubdtype(arr.dtype, np.bool_):
        raise TypeError(
            f"expected the binary array to be boolean dtype but got {arr.dtype}"
        )

    # Create a polygon of the binary tissue mask.
    scale: tuple[float, float] = (
        slide.dimensions[0] / thumb.size[0],
        slide.dimensions[1] / thumb.size[1],
    )
    polygon = get_multipolygon_from_binary_arr(arr.astype("uint8") * 255, scale=scale)

    # Get the coordinates of patches inside the tissue polygon.
    slide_width, slide_height = slide.dimensions
    half_patch_size = round(patch_size / 2)

    # Nx4 --> N x (minx, miny, width, height)
    coords = get_nonoverlapping_patch_coordinates_within_polygon(
        slide_width=slide_width,
        slide_height=slide_height,
        patch_size=patch_size,
        half_patch_size=half_patch_size,
        polygon=polygon,
    )
    logger.info(f"Found {len(coords)} patches within tissue")

    return coords


def save_hdf5(
    path: str | Path,
    coords: npt.NDArray[np.int_],
    patch_size: int,
    patch_spacing_um_px: float,
    compression: str | None = "gzip",
):
    """Write patch coordinates to HDF5 file.

    This is designed to be interoperable with HDF5 files created by CLAM.
    """
    logger.info(f"Writing coordinates to disk: {path}")
    logger.info(f"Coordinates have shape {coords.shape}")

    if coords.ndim != 2:
        raise ValueError(f"coords must have 2 dimensions but got {coords.ndim}")
    if coords.shape[1] != 2:
        raise ValueError(f"length of second axis must be 2 but got {coords.shape[1]}")

    with h5py.File(path, "w") as f:
        dset = f.create_dataset("/coords", data=coords, compression=compression)
        dset.attrs["patch_size"] = patch_size
        dset.attrs["patch_spacing_um_px"] = patch_spacing_um_px
