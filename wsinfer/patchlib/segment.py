"""Segment thumbnail of a whole slide image."""

from __future__ import annotations

import cv2 as cv
import numpy as np
import numpy.typing as npt
from skimage.morphology import binary_closing
from skimage.morphology import remove_small_holes
from skimage.morphology import remove_small_objects


def segment_tissue(
    im_arr: npt.NDArray[np.uint8],
    median_filter_size: int = 7,
    binary_threshold: int = 7,
    closing_kernel_size: int = 6,
    min_object_size_px: int = 512,
    min_hole_size_px: int = 1024,
) -> npt.NDArray[np.bool_]:
    """Create a binary tissue mask from an image.

    Parameters
    ----------
    im_arr : array-like
        RGB image array (uint8).
    """
    im_arr = np.asarray(im_arr)
    assert im_arr.ndim == 3
    assert im_arr.shape[2] == 3

    # Convert to HSV color space.
    im_arr = cv.cvtColor(im_arr, cv.COLOR_RGB2HSV)
    im_arr = im_arr[:, :, 1]  # Keep saturation channel only.

    # Use median blurring to smooth the image.
    if median_filter_size <= 1 or median_filter_size % 2 == 0:
        raise ValueError(
            "median_filter_size must be greater than 1 and odd, but got"
            f" {median_filter_size}"
        )

    # We use opencv here instead of PIL because opencv is _much_ faster. We use skimage
    # further down for artifact removal (hole filling, object removal) because skimage
    # provides easy to use methods for those.
    im_arr = cv.medianBlur(im_arr, median_filter_size)

    # Binarize image.
    _, im_arr = cv.threshold(
        im_arr, thresh=binary_threshold, maxval=255, type=cv.THRESH_BINARY
    )

    # Convert to boolean dtype. This helps with static type analysis because at this
    # point, im_arr is a uint8 array.
    im_arr_binary = im_arr > 0

    # Closing. This removes small holes. It might not be entirely necessary because
    # we have hole removal below.
    im_arr_binary = binary_closing(
        im_arr_binary, footprint=np.ones((closing_kernel_size, closing_kernel_size))
    )

    # Remove small objects.
    im_arr_binary = remove_small_objects(im_arr_binary, min_size=min_object_size_px)

    # Remove small holes.
    im_arr_binary = remove_small_holes(im_arr_binary, area_threshold=min_hole_size_px)

    return im_arr_binary
