"""Create a dense grid of patch coordinates. This does *not* create a tissue mask."""

import itertools
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import tiffslide

from wsinfer.wsi import get_avg_mpp


def _get_dense_grid(
    slide, orig_patch_size: int, patch_spacing_um_px: float
) -> Tuple[np.ndarray, int]:
    mpp = get_avg_mpp(slide)
    patch_size = orig_patch_size * patch_spacing_um_px / mpp
    patch_size = round(patch_size)
    step_size = patch_size  # non-overlapping patches
    oslide = tiffslide.TiffSlide(slide)
    cols, rows = oslide.level_dimensions[0]
    xs = range(0, cols, step_size)
    ys = range(0, rows, step_size)
    # List of (x, y) coordinates.
    return np.asarray(list(itertools.product(xs, ys))), patch_size


def create_grid_and_save(
    slide, results_dir, orig_patch_size: int, patch_spacing_um_px: float
):
    """Create dense grid of (x,y) coordinates and save to HDF5.

    This is similar to the CLAM coordinate code but does not use a tissue mask.
    """
    slide = Path(slide)
    results_dir = Path(results_dir)
    hdf5_path = results_dir / "patches" / f"{slide.stem}.h5"
    hdf5_path.parent.mkdir(exist_ok=True)
    coords, patch_size = _get_dense_grid(
        slide=slide,
        orig_patch_size=orig_patch_size,
        patch_spacing_um_px=patch_spacing_um_px,
    )
    with h5py.File(hdf5_path, "w") as f:
        dset = f.create_dataset("/coords", data=coords, compression="gzip")
        dset.attrs["name"] = str(hdf5_path.stem)
        dset.attrs["patch_level"] = 0
        dset.attrs["patch_size"] = patch_size
        dset.attrs["save_path"] = str(hdf5_path.parent)


def create_grid_and_save_multi_slides(
    wsi_dir, results_dir, orig_patch_size: int, patch_spacing_um_px: float
):
    wsi_dir = Path(wsi_dir)
    slides = list(wsi_dir.glob("*"))
    if not slides:
        raise FileNotFoundError("no slides found")

    for slide in slides:
        create_grid_and_save(
            slide=slide,
            results_dir=results_dir,
            orig_patch_size=orig_patch_size,
            patch_spacing_um_px=patch_spacing_um_px,
        )
