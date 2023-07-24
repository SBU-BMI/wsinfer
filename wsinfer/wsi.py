from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path
from typing import Literal
from typing import Protocol
from typing import overload

import tifffile
from PIL import Image

from .errors import CannotReadSpacing
from .errors import DuplicateFilePrefixesFound
from .errors import NoBackendException

logger = logging.getLogger(__name__)


try:
    import openslide

    HAS_OPENSLIDE = True
    logger.debug("Imported openslide")
except Exception as err:
    HAS_OPENSLIDE = False
    logger.debug(f"Unable to import openslide due to error: {err}")

try:
    import tiffslide

    HAS_TIFFSLIDE = True
    logger.debug("Imported tiffslide")
except Exception as err:
    HAS_TIFFSLIDE = False
    logger.debug(f"Unable to import tiffslide due to error: {err}")


@overload
def set_backend(name: Literal["openslide"]) -> type[openslide.OpenSlide]:
    ...


@overload
def set_backend(name: Literal["tiffslide"]) -> type[tiffslide.TiffSlide]:
    ...


def set_backend(
    name: Literal["openslide"] | Literal["tiffslide"],
) -> type[tiffslide.TiffSlide] | type[openslide.OpenSlide]:
    global WSI
    if name not in ["openslide", "tiffslide"]:
        raise ValueError(f"Unknown backend: {name}")
    logger.info(f"Setting backend to {name}")
    if name == "openslide":
        WSI = openslide.OpenSlide
    elif name == "tiffslide":
        WSI = tiffslide.TiffSlide
    else:
        raise ValueError(f"Unknown backend: {name}")
    return WSI


# Set the slide backend based on the environment.
WSI: type[openslide.OpenSlide] | type[tiffslide.TiffSlide]
if HAS_OPENSLIDE:
    WSI = set_backend("openslide")
elif HAS_TIFFSLIDE:
    WSI = set_backend("tiffslide")
else:
    raise NoBackendException("No backend found! Please install openslide or tiffslide")


# For typing an object that has a method `read_region`.


class CanReadRegion(Protocol):
    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image.Image:
        pass


def _get_mpp_openslide(slide_path: str | Path) -> tuple[float, float]:
    """Read MPP using OpenSlide.

    Parameters
    ----------
    slide_path : str or Path
        The path to the whole slide image.

    Returns
    -------
    mppx, mppy
        Two floats representing the micrometers per pixel in x and y dimensions.

    Raises
    ------
    CannotReadSpacing if spacing cannot be read from the whole slide iamge.
    """
    logger.debug("Attempting to read MPP using OpenSlide")
    slide = openslide.OpenSlide(slide_path)
    mppx: float | None = None
    mppy: float | None = None

    if (
        openslide.PROPERTY_NAME_MPP_X in slide.properties
        and openslide.PROPERTY_NAME_MPP_Y in slide.properties
    ):
        logger.debug(
            "Properties of the OpenSlide object contains keys"
            f" {openslide.PROPERTY_NAME_MPP_X} and {openslide.PROPERTY_NAME_MPP_Y}"
        )
        mppx = slide.properties[openslide.PROPERTY_NAME_MPP_X]
        mppy = slide.properties[openslide.PROPERTY_NAME_MPP_Y]
        logger.debug(
            f"Value of {openslide.PROPERTY_NAME_MPP_X} is {mppx} and value"
            f" of {openslide.PROPERTY_NAME_MPP_Y} is {mppy}"
        )
        if mppx is not None and mppy is not None:
            try:
                logger.debug("Attempting to convert these MPP strings to floats")
                mppx = float(mppx)
                mppy = float(mppy)
                return mppx, mppy
            except Exception as err:
                logger.debug(f"Exception caught while converting to float: {err}")
    elif (
        "tiff.ResolutionUnit" in slide.properties
        and "tiff.XResolution" in slide.properties
        and "tiff.YResolution" in slide.properties
    ):
        logger.debug("Attempting to read spacing using openslide and tiff tags")
        resunit = slide.properties["tiff.ResolutionUnit"].lower()
        if resunit not in {"millimeter", "centimeter", "cm", "inch"}:
            raise CannotReadSpacing(f"unknown resolution unit: '{resunit}'")
        scale = {
            "inch": 25400.0,
            "centimeter": 10000.0,
            "cm": 10000.0,
            "millimeter": 1000.0,
        }.get(resunit, None)

        x_resolution = float(slide.properties["tiff.XResolution"])
        y_resolution = float(slide.properties["tiff.YResolution"])

        if scale is not None:
            try:
                mpp_x = scale / x_resolution
                mpp_y = scale / y_resolution
                return mpp_x, mpp_y
            except ArithmeticError as err:
                raise CannotReadSpacing(
                    f"error in math {scale} / {x_resolution}"
                    f" or {scale} / {y_resolution}"
                ) from err
        else:
            raise CannotReadSpacing()

    else:
        logger.debug(
            "Properties of the OpenSlide object does not contain keys"
            f" {openslide.PROPERTY_NAME_MPP_X} and {openslide.PROPERTY_NAME_MPP_Y}"
        )
    raise CannotReadSpacing()


def _get_mpp_tiffslide(
    slide_path: str | Path,
) -> tuple[float, float]:
    """Read MPP using TiffSlide."""
    slide = tiffslide.TiffSlide(slide_path)
    mppx: float | None = None
    mppy: float | None = None
    if (
        tiffslide.PROPERTY_NAME_MPP_X in slide.properties
        and tiffslide.PROPERTY_NAME_MPP_Y in slide.properties
    ):
        mppx = slide.properties[tiffslide.PROPERTY_NAME_MPP_X]
        mppy = slide.properties[tiffslide.PROPERTY_NAME_MPP_Y]
        if mppx is None or mppy is None:
            raise CannotReadSpacing()
        else:
            try:
                mppx = float(mppx)
                mppy = float(mppy)
                return mppx, mppy
            except Exception as err:
                raise CannotReadSpacing() from err
    raise CannotReadSpacing()


# Modified from
# https://github.com/bayer-science-for-a-better-life/tiffslide/blob/8bea5a4c8e1429071ade6d4c40169ce153786d19/tiffslide/tiffslide.py#L712-L745
def _get_mpp_tifffile(slide_path: str | Path) -> tuple[float, float]:
    """Read MPP using Tifffile."""
    with tifffile.TiffFile(slide_path) as tif:
        series0 = tif.series[0]
        page0 = series0[0]
        try:
            resolution_unit = page0.tags["ResolutionUnit"].value
            x_resolution = Fraction(*page0.tags["XResolution"].value)
            y_resolution = Fraction(*page0.tags["YResolution"].value)
        except KeyError as err:
            raise CannotReadSpacing() from err

        RESUNIT = tifffile.TIFF.RESUNIT
        scale = {
            RESUNIT.INCH: 25400.0,
            RESUNIT.CENTIMETER: 10000.0,
            RESUNIT.MILLIMETER: 1000.0,
            RESUNIT.MICROMETER: 1.0,
            RESUNIT.NONE: None,
        }.get(resolution_unit, None)
        if scale is not None:
            try:
                mpp_x = scale / x_resolution
                mpp_y = scale / y_resolution
                return mpp_x, mpp_y
            except ArithmeticError as err:
                raise CannotReadSpacing() from err
    raise CannotReadSpacing()


def get_avg_mpp(slide_path: Path | str) -> float:
    """Return the average MPP of a whole slide image.

    The value is in units of micrometers per pixel and is
    the average of the X and Y dimensions.

    Raises
    ------
    CannotReadSpacing if the spacing cannot be read.
    """

    mppx: float
    mppy: float

    if HAS_OPENSLIDE:
        try:
            mppx, mppy = _get_mpp_openslide(slide_path)
            return (mppx + mppy) / 2
        except CannotReadSpacing:
            # At this point, we want to continue to other implementations.
            pass
    if HAS_TIFFSLIDE:
        try:
            mppx, mppy = _get_mpp_tiffslide(slide_path)
            return (mppx + mppy) / 2
        except CannotReadSpacing:
            # Our last hope to read the mpp is tifffile.
            pass
    try:
        mppx, mppy = _get_mpp_tifffile(slide_path)
        return (mppx + mppy) / 2
    except CannotReadSpacing:
        pass

    raise CannotReadSpacing(slide_path)


def _validate_wsi_directory(wsi_dir: str | Path) -> None:
    """Validate a directory of whole slide images."""
    wsi_dir = Path(wsi_dir)
    maybe_slides = sorted(wsi_dir.glob("*"))
    uniq_stems = set(p.stem for p in maybe_slides)
    if len(uniq_stems) != len(maybe_slides):
        raise DuplicateFilePrefixesFound(
            "A slide with the same prefix but different extensions has been found"
            " (like slide.svs and slide.tif). Slides must have unique prefixes."
        )
