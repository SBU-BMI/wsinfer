from __future__ import annotations

import itertools
from typing import Sequence

import cv2 as cv
import numpy as np
import numpy.typing as npt
from shapely import MultiPolygon
from shapely import Point
from shapely import Polygon
from shapely import STRtree


def get_multipolygon_from_binary_arr(
    arr: npt.NDArray[np.int_], scale: tuple[float, float] | None = None
) -> MultiPolygon:
    """Create a Shapely Polygon from a binary array."""
    # Find contours and hierarchy
    contours: Sequence[npt.NDArray[np.int_]]
    hierarchy: npt.NDArray[np.int_]
    contours, hierarchy = cv.findContours(arr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy.squeeze(0)

    contours_unscaled = contours
    if scale is not None:
        # To broadcast with contour coordinates.
        scale_arr = np.array(scale).reshape(1, 1, 2)
        contours = tuple(c * scale_arr for c in contours_unscaled)
        del scale_arr

    # https://stackoverflow.com/a/75510437/5666087
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

    # Call the function with an initial empty polygon and start from contour 0
    polygon = merge_polygons(MultiPolygon(), 0, True)
    return polygon


def get_nonoverlapping_patch_coordinates_within_polygon(
    slide_width: int,
    slide_height: int,
    patch_size: int,
    half_patch_size: int,
    polygon: Polygon,
) -> npt.NDArray[np.int_]:
    """Get coordinates of patches within a polygon.

    Returns
    -------
    Array with shape (N, 4), where N is the number of tiles. Each row in this array
    contains the coordinates of a tile: (minx, miny, width, height).
    """

    # Make an array of Nx2, where each row is (x, y) centroid of the patch.
    tile_centroids_arr: npt.NDArray[np.int_] = np.array(
        list(
            itertools.product(
                range(0 + half_patch_size, slide_width - half_patch_size, patch_size),
                range(0 + half_patch_size, slide_height - half_patch_size, patch_size),
            )
        )
    )

    # NOTE: this line seems to be the bottleneck of the functions.
    tile_centroids_poly = [Point(c) for c in tile_centroids_arr]

    # Query which centroids are inside the polygon.
    tree = STRtree(tile_centroids_poly)
    centroid_indexes_in_polygon = tree.query(polygon, predicate="contains")
    tile_centroids_in_polygon = tile_centroids_arr[centroid_indexes_in_polygon]

    # Transform the centroids to the upper-left point (x, y).
    tile_minx_miny_in_polygon = tile_centroids_in_polygon - half_patch_size

    # Add the patch size to the coordinate array.
    # patch_size_arr = np.full_like(tile_minx_miny_in_polygon, patch_size)
    # coords = np.concatenate((tile_minx_miny_in_polygon, patch_size_arr), axis=1)

    return tile_minx_miny_in_polygon
