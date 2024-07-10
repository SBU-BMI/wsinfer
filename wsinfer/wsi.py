from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path
from typing import Literal
from typing import Protocol
from typing import overload

import tifffile
from PIL import Image

from .errors import BackendNotAvailable
from .errors import CannotReadSpacing
from .errors import DuplicateFilePrefixesFound
from .errors import NoBackendException

logger = logging.getLogger(__name__)

_BACKEND: str = "tiffslide"

_allowed_backends = {"openslide", "tiffslide"}

try:
    import openslide

    # Test that OpenSlide object exists. If it doesn't, an error will be thrown and
    # caught. For some reason, it is possible that openslide-python can be installed
    # but the OpenSlide object (and other openslide things) are not available.
    openslide.OpenSlide  # noqa: B018
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

if not HAS_TIFFSLIDE and not HAS_OPENSLIDE:
    raise NoBackendException(
        "No backend is available. Please install openslide or tiffslide."
    )


def set_backend(name: str) -> None:
    global _BACKEND
    if name not in _allowed_backends:
        raise ValueError(f"Unknown backend: '{name}'")
    if name == "openslide" and not HAS_OPENSLIDE:
        raise BackendNotAvailable(
            "OpenSlide is not available. Please install the OpenSlide compiled"
            " library and the Python package 'openslide-python'."
            " See https://openslide.org/ for more information."
        )
    elif name == "tiffslide":
        if not HAS_TIFFSLIDE:
            raise BackendNotAvailable(
                "TiffSlide is not available. Please install 'tiffslide'."
            )

    logger.debug(f"Set backend to {name}")

    _BACKEND = name


def get_wsi_cls() -> type[openslide.OpenSlide] | type[tiffslide.TiffSlide]:
    if _BACKEND not in _allowed_backends:
        raise ValueError(
            f"Unknown backend: '{_BACKEND}'. Please contact the developer!"
        )
    if _BACKEND == "openslide":
        return openslide.OpenSlide  # type: ignore
    elif _BACKEND == "tiffslide":
        return tiffslide.TiffSlide
    else:
        raise ValueError("Contact the developer, slide backend not known")


# Set the slide backend based on the environment.
# Prioritize TiffSlide if the user has it installed.
if HAS_TIFFSLIDE:
    set_backend("tiffslide")
elif HAS_OPENSLIDE:
    set_backend("openslide")
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
    if not HAS_OPENSLIDE:
        logger.critical(
            "Cannot read MPP with OpenSlide because OpenSlide is not available"
        )
        raise CannotReadSpacing()
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
    if not HAS_TIFFSLIDE:
        logger.critical(
            "Cannot read MPP with TiffSlide because TiffSlide is not available"
        )
        raise CannotReadSpacing()

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
        if not isinstance(page0, tifffile.TiffPage):
            raise CannotReadSpacing("not a tifffile.TiffPage instance")
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
    maybe_slides = [p for p in wsi_dir.iterdir() if p.is_file()]
    uniq_stems = set(p.stem for p in maybe_slides)
    if len(uniq_stems) != len(maybe_slides):
        raise DuplicateFilePrefixesFound(
            "A slide with the same prefix but different extensions has been found"
            " (like slide.svs and slide.tif). Slides must have unique prefixes."
        )
