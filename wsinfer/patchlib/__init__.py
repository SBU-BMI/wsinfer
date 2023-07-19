from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2 as cv
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

MASKS_DIR = "masks"
PATCHES_DIR = "patches"


def segment_and_patch_one_slide(
    slide_path: str | Path,
    save_dir: str | Path,
    patch_size_px: int,
    patch_spacing_um_px: float,
    thumbsize: tuple[int, int] = (2048, 2048),
    median_filter_size: int = 7,
    binary_threshold: int = 7,
    closing_kernel_size: int = 6,
    min_object_size_um2: float = 200**2,
    min_hole_size_um2: float = 190**2,
) -> None:
    """Get patch coordinates."""

    save_dir = Path(save_dir).resolve()
    slide_path = Path(slide_path).resolve()
    slide_prefix = slide_path.stem

    logger.info(f"Segmenting and patching slide {slide_path}")
    logger.info(f"Using prefix as slide ID: {slide_prefix}")

    patch_path = save_dir / PATCHES_DIR / f"{slide_prefix}.h5"
    mask_path = save_dir / MASKS_DIR / f"{slide_prefix}.jpg"

    # End early if outputs exist.
    if patch_path.exists() and mask_path.exists():
        logger.info("Patch output and mask output files already exist")
        logger.info(f"patch_path={patch_path}")
        logger.info(f"mask_path={mask_path}")
        return None

    slide = WSI(slide_path)
    mpp = get_avg_mpp(slide_path)
    logger.info(f"Slide has WxH {slide.dimensions} and MPP={mpp}")

    logger.info(
        f"Requested patch size of {patch_size_px} px at {patch_spacing_um_px} um/px"
    )
    logger.info(
        f"Scaling patch size by {patch_spacing_um_px / mpp} for patch coordinates at"
        f" level 0 (MPP={mpp}) | patch_spacing_um_px / mpp"
        f" ({patch_spacing_um_px} / {mpp})"
    )
    patch_size = round(patch_size_px * patch_spacing_um_px / mpp)
    logger.info(f"Final patch size is {patch_size}")

    # Segment tissue into a binary mask.
    if len(thumbsize) != 2:
        raise ValueError(f"Length of 'thumbsize' must be 2 but got {len(thumbsize)}")
    thumb: Image.Image = slide.get_thumbnail(thumbsize)
    # TODO: allow the min hole size and min object size to be set in physical units.

    # thumb has ~12 MPP.
    thumb_mpp = (mpp * (np.array(slide.dimensions) / thumb.size)).mean()
    logger.info(f"Thumbnail has WxH {thumb.size} and MPP={thumb_mpp}")
    thumb_mpp_squared: float = thumb_mpp**2

    # (pixels2 / micron2) * micron2 = pixels2
    min_object_size_px: int = round(min_object_size_um2 / thumb_mpp_squared)
    min_hole_size_px: int = round(min_hole_size_um2 / thumb_mpp_squared)

    logger.info(
        f"Transformed minimum object size to {min_object_size_px} pixel area in"
        " thumbnail"
    )
    logger.info(
        f"Transformed minimum hole size to {min_hole_size_px} pixel area in thumbnail"
    )

    arr = segment_tissue(
        np.asarray(thumb),
        median_filter_size=median_filter_size,
        binary_threshold=binary_threshold,
        closing_kernel_size=closing_kernel_size,
        min_object_size_px=min_object_size_px,
        min_hole_size_px=min_hole_size_px,
    )
    if not np.issubdtype(arr.dtype, np.bool_):
        raise TypeError(
            f"expected the segmentation array to be boolean dtype but got {arr.dtype}"
        )

    # Create a polygon of the binary tissue mask.
    scale: tuple[float, float] = (
        slide.dimensions[0] / thumb.size[0],
        slide.dimensions[1] / thumb.size[1],
    )
    polygon, contours, hierarchy = get_multipolygon_from_binary_arr(
        arr.astype("uint8") * 255, scale=scale
    )

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

    # Save coordinates to HDF5.
    patch_path.parent.mkdir(exist_ok=True, parents=True)
    save_hdf5(
        path=patch_path,
        coords=coords,
        patch_size=patch_size,
        patch_spacing_um_px=patch_spacing_um_px,
        compression="gzip",
    )

    # Save thumbnail with drawn contours.
    logger.info(f"Writing tissue thumbnail with contours to disk: {mask_path}")
    mask_path.parent.mkdir(exist_ok=True, parents=True)
    img = draw_contours_on_thumbnail(thumb, contours=contours, hierarchy=hierarchy)
    img.thumbnail((1024, 1024), resample=Image.Resampling.LANCZOS)
    img.save(mask_path)

    return None


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
        dset.attrs["patch_level"] = 0
        dset.attrs["patch_spacing_um_px"] = patch_spacing_um_px


def draw_contours_on_thumbnail(
    thumb: Image.Image,
    contours: Sequence[npt.NDArray[np.int_]],
    hierarchy: npt.NDArray[np.int_],
) -> Image.Image:
    assert hierarchy.ndim == 3
    assert hierarchy.shape[0] == 1
    assert hierarchy.shape[2] == 4
    assert len(contours) == hierarchy.shape[1]

    # We assume the hierarchy was made with RETR_CCOMP.
    contour_is_external = (hierarchy[0, :, 3] < 0).tolist()
    external = [c for c, external in zip(contours, contour_is_external) if external]
    hole = [c for c, external in zip(contours, contour_is_external) if not external]

    img = np.array(thumb)
    # Colors are BGR.
    cv.drawContours(img, external, -1, (0, 255, 255), 7)
    cv.drawContours(img, hole, -1, (255, 255, 0), 7)

    return Image.fromarray(img).convert("RGB")


def segment_and_patch_directory_of_slides(
    wsi_dir: str | Path,
    save_dir: str | Path,
    patch_size_px: int,
    patch_spacing_um_px: float,
    thumbsize: tuple[int, int] = (2048, 2048),
    median_filter_size: int = 7,
    binary_threshold: int = 7,
    closing_kernel_size: int = 6,
    min_object_size_um2: float = 200**2,
    min_hole_size_um2: float = 190**2,
) -> None:
    """Get patch coordinates. for a directory of slides."""

    wsi_dir = Path(wsi_dir)
    slide_paths = sorted(wsi_dir.glob("*"))

    # NOTE: we could use multi-processing here but then the logs would get
    # discombobulated.
    for i, slide_path in enumerate(slide_paths):
        print(f"Slide {i+1} of {len(slide_paths)} ({(i+1)/len(slide_paths):.2%})")
        try:
            segment_and_patch_one_slide(
                slide_path=slide_path,
                save_dir=save_dir,
                patch_size_px=patch_size_px,
                patch_spacing_um_px=patch_spacing_um_px,
                thumbsize=thumbsize,
                median_filter_size=median_filter_size,
                binary_threshold=binary_threshold,
                closing_kernel_size=closing_kernel_size,
                min_object_size_um2=min_object_size_um2,
                min_hole_size_um2=min_hole_size_um2,
            )
        except Exception as e:
            logger.error(f"Failed to segment and patch slide\n{slide_path}", exc_info=e)

    return None
