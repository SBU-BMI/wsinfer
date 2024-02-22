from __future__ import annotations

import itertools
import logging
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import Iterator
from typing import Sequence
from typing import cast as type_cast

import cv2 as cv
import numpy as np
import numpy.typing as npt
from shapely import MultiPolygon
from shapely import Point
from shapely import Polygon
from shapely import STRtree

logger = logging.getLogger(__name__)


@contextmanager
def temporary_recursion_limit(limit: int) -> Iterator[None]:
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(limit)
        yield
    finally:
        sys.setrecursionlimit(old_limit)


def get_multipolygon_from_binary_arr(
    arr: npt.NDArray[np.int_], scale: tuple[float, float] | None = None
) -> tuple[MultiPolygon, Sequence[npt.NDArray[np.int_]], npt.NDArray[np.int_]] | None:
    """Create a Shapely Polygon from a binary array.

    Parameters
    ----------
    arr : array
        Binary array where non-zero values indicate presence of tissue.
    scale : tuple of two floats, optional
        If specified, this is the factor by which coordinates are multiplied to recover
        the coordinates at the base resolution of the whole slide image.

    Returns
    -------
    polygon
        A shapely `MultiPolygon` object representing tissue regions.
    contours
        A sequence of arrays representing unscaled contours of tissue.
    hierarchy
        An array of the hierarchy of contours.
    """
    # Find contours and hierarchy
    contours: Sequence[npt.NDArray]
    hierarchy: npt.NDArray | None
    contours, hierarchy = cv.findContours(arr, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    hierarchy = hierarchy.squeeze(0)

    logger.info(f"Detected {len(contours)} contours")
    contours_unscaled = contours
    if scale is not None:
        logger.info(
            "Scaling contours to slide-level (level=0) coordinate space by multiplying"
            f" by {scale}"
        )
        # Reshape to broadcast with contour coordinates.
        scale_arr: npt.NDArray = np.array(scale).reshape(1, 1, 2)
        contours = tuple(c * scale_arr for c in contours_unscaled)
        del scale_arr

    # From https://stackoverflow.com/a/75510437/5666087
    def merge_polygons(polygon: MultiPolygon, idx: int, add: bool) -> MultiPolygon:
        """
        polygon: Main polygon to which a new polygon is added
        idx: Index of contour
        add: If this contour should be added (True) or subtracted (False)
        """

        # Get contour from global list of contours
        contour = np.squeeze(contours[idx])

        # cv2.findContours() sometimes returns a single point -> skip this case
        if len(contour) > 2:
            # Convert contour to shapely polygon
            new_poly = Polygon(contour)

            # Not all polygons are shapely-valid (self intersection, etc.)
            if not new_poly.is_valid:
                # Convert invalid polygon to valid
                new_poly = new_poly.buffer(0)

            # Merge new polygon with the main one
            if add:
                polygon = polygon.union(new_poly)
            else:
                polygon = polygon.difference(new_poly)

        # Check if current polygon has a child
        if hierarchy is None:
            raise NotImplementedError()
        child_idx = hierarchy[idx][2]
        if child_idx >= 0:
            # Call this function recursively, negate `add` parameter
            polygon = merge_polygons(polygon, child_idx, not add)

        # Check if there is some next polygon at the same hierarchy level
        next_idx = hierarchy[idx][0]
        if next_idx >= 0:
            # Call this function recursively
            polygon = merge_polygons(polygon, next_idx, add)

        return polygon

    temp_limit = max(sys.getrecursionlimit(), len(contours))
    with temporary_recursion_limit(temp_limit):
        # Call the function with an initial empty polygon and start from contour 0
        polygon = merge_polygons(MultiPolygon(), 0, True)

    if TYPE_CHECKING:
        hierarchy = type_cast(npt.NDArray[np.int_], hierarchy)
        contours_unscaled = type_cast(Sequence[npt.NDArray[np.int_]], contours_unscaled)

    # Add back the axis in hierarchy because we squeezed it before.
    return polygon, contours_unscaled, hierarchy[np.newaxis]


def get_nonoverlapping_patch_coordinates_within_polygon(
    slide_width: int,
    slide_height: int,
    patch_size: int,
    half_patch_size: int,
    polygon: Polygon,
) -> npt.NDArray[np.int_]:
    """Get coordinates of patches within a polygon.

    Parameters
    ----------
    slide_width : int
        The width of the slide in pixels at base resolution.
    slide_height : int
        The height of the slide in pixels at base resolution.
    patch_size : int
        The size of a patch in pixels.
    half_patch_size : int
        Half of the length of a patch in pixels.
    polygon : Polygon
        A shapely Polygon representing the presence of tissue.

    Returns
    -------
    coordinates
        Array with shape (N, 2), where N is the number of tiles. Each row in this array
        contains the coordinates of the top-left of a tile: (minx, miny).
    """

    # Make an array of Nx2, where each row is (x, y) centroid of the patch.
    tile_centroids_arr: npt.NDArray[np.int_] = np.array(
        list(
            itertools.product(
                range(0 + half_patch_size, slide_width, patch_size),
                range(0 + half_patch_size, slide_height, patch_size),
            )
        )
    )

    tile_centroids_poly = [Point(c) for c in tile_centroids_arr]

    # Query which centroids are inside the polygon.
    tree = STRtree(tile_centroids_poly)
    centroid_indexes_in_polygon: npt.NDArray[np.int_] = tree.query(
        polygon, predicate="contains"
    )

    # Sort so x and y are in ascending order (and y changes most rapidly).
    centroid_indexes_in_polygon.sort()
    tile_centroids_in_polygon = tile_centroids_arr[centroid_indexes_in_polygon]

    # Transform the centroids to the upper-left point (x, y).
    tile_minx_miny_in_polygon = tile_centroids_in_polygon - half_patch_size

    return tile_minx_miny_in_polygon
