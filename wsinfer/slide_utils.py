from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tiffslide
import tifffile

from .errors import CannotReadSpacing

PathType = Union[str, Path]


def _get_mpp_tiffslide(slide_path: PathType) -> Tuple[Optional[float], Optional[float]]:
    slide = tiffslide.TiffSlide(slide_path)
    mppx: Optional[float] = None
    mppy: Optional[float] = None
    if (
        tiffslide.PROPERTY_NAME_MPP_X in slide.properties
        and tiffslide.PROPERTY_NAME_MPP_Y in slide.properties
    ):
        mppx = slide.properties[tiffslide.PROPERTY_NAME_MPP_X]
        mppy = slide.properties[tiffslide.PROPERTY_NAME_MPP_Y]
        if mppx is None or mppy is None:
            raise ValueError(
                "Cannot infer slide spacing because MPPX or MPPY is None:"
                f" {mppx} and {mppy}"
            )
        else:
            mppx = float(mppx)
            mppy = float(mppy)

    return mppx, mppy


def _get_biggest_series(tif: tifffile.TiffFile) -> int:
    max_area: int = 0
    max_index: Optional[int] = None
    for index, s in enumerate(tif.series):
        area = np.prod(s.shape)
        if area > max_area and "X" in s.axes and "Y" in s.axes:
            max_area = area
            max_index = index
    if max_index is None:
        raise ValueError("Cannot find largest series in the slide")
    return max_index


def _get_mpp_tiff_tiffslide(
    slide_path: PathType,
) -> Tuple[Optional[float], Optional[float]]:
    resunit_to_microns = {"inch": 25400, "in": 25400, "centimeter": 10000, "cm": 10000}
    um_x: Optional[float] = None
    um_y: Optional[float] = None

    slide = tiffslide.TiffSlide(slide_path)

    if (
        "tiff.ResolutionUnit" in slide.properties
        and "tiff.XResolution" in slide.properties
        and "tiff.YResolution" in slide.properties
    ):
        unit = resunit_to_microns[slide.properties["tiff.ResolutionUnit"]]
        xres = float(slide.properties["tiff.XResolution"])
        yres = float(slide.properties["tiff.YResolution"])
        if xres >= 100:
            um_x = unit / xres
        if yres >= 100:
            um_y = unit / yres
    return um_x, um_y


def _get_mpp_tiff_tifffile(
    slide_path: PathType,
) -> Tuple[Optional[float], Optional[float]]:
    # Enum ResolutionUnit value to the number of micrometers in that unit.
    # 2: inch (25,400 microns in an inch)
    # 3: centimeter (10,000 microns in a cm)
    resunit_to_microns = {2: 25400, 3: 10000}
    um_x: Optional[float] = None
    um_y: Optional[float] = None

    with tifffile.TiffFile(slide_path) as tif:
        biggest_series = _get_biggest_series(tif)
        s = tif.series[biggest_series]
        page = s.pages[0]
        unit = resunit_to_microns[page.tags["ResolutionUnit"].value.real]
        if page.tags["XResolution"].value[1] >= 100:
            um_x = (
                unit
                * page.tags["XResolution"].value[1]
                / page.tags["XResolution"].value[0]
            )
        if page.tags["YResolution"].value[1] >= 100:
            um_y = (
                unit
                * page.tags["YResolution"].value[1]
                / page.tags["YResolution"].value[0]
            )
    return um_x, um_y


def get_avg_mpp(slide_path: PathType) -> float:
    """Return the average MPP of a whole slide image."""

    mppx, mppy = _get_mpp_tiffslide(slide_path)
    if mppx is not None and mppy is not None:
        return (mppx + mppy) / 2

    # Try tiffslide with tiff tags now.
    mppx, mppy = _get_mpp_tiff_tiffslide(slide_path)
    if mppx is not None and mppy is not None:
        return (mppx + mppy) / 2

    # Try tifffile now.
    mppx, mppy = _get_mpp_tiff_tifffile(slide_path)
    if mppx is not None and mppy is not None:
        return (mppx + mppy) / 2

    raise CannotReadSpacing(f"Could not read the spacing of slide {slide_path}")
