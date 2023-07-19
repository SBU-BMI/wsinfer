from __future__ import annotations

from pathlib import Path
from typing import Callable
from typing import Sequence

import h5py
import numpy as np
import torch
from PIL import Image

from wsinfer.wsi import WSI


def _read_patch_coords(path: str | Path) -> np.ndarray:
    """Read HDF5 file of patch coordinates are return numpy array.

    Returned array has shape (num_patches, 4). Each row has values
    [minx, miny, width, height].
    """
    with h5py.File(path, mode="r") as f:
        coords = f["/coords"][()]
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


def _filter_patches_in_rois(
    *, geojson_path: str | Path, coords: np.ndarray
) -> np.ndarray:
    """Keep the patches that intersect the ROI(s).

    Parameters
    ----------
    geojson_path : str, Path
        Path to the GeoJSON file that encodes the points of the ROI(s).
    coords : ndarray
        Two-dimensional array where each row has minx, miny, width, height.

    Returns
    -------
    ndarray of filtered coords.
    """
    import geojson
    from shapely import STRtree
    from shapely.geometry import box
    from shapely.geometry import shape

    with open(geojson_path) as f:
        geo = geojson.load(f)
    if not geo.is_valid:
        raise ValueError("GeoJSON of ROI is not valid")
    for roi in geo["features"]:
        assert roi.is_valid, "an ROI geometry is not valid"
    geoms_rois = [shape(roi["geometry"]) for roi in geo["features"]]
    coords_orig = coords.copy()
    coords = coords.copy()
    coords[:, 2] += coords[:, 0]  # Calculate maxx.
    coords[:, 3] += coords[:, 1]  # Calculate maxy.
    boxes = [box(*coords[idx]) for idx in range(coords.shape[0])]
    tree = STRtree(boxes)
    _, intersecting_ids = tree.query(geoms_rois, predicate="intersects")
    intersecting_ids = np.sort(np.unique(intersecting_ids))
    return coords_orig[intersecting_ids]


class WholeSlideImagePatches(torch.utils.data.Dataset):
    """Dataset of one whole slide image.

    This object retrieves patches from a whole slide image on the fly.

    Parameters
    ----------
    wsi_path : str, Path
        Path to whole slide image file.
    patch_path : str, Path
        Path to npy file with coordinates of input image.
    um_px : float
        Scale of the resulting patches. For example, 0.5 for ~20x magnification.
    patch_size : int
        The size of patches in pixels.
    transform : callable, optional
        A callable to modify a retrieved patch. The callable must accept a
        PIL.Image.Image instance and return a torch.Tensor.
    roi_path : str, Path, optional
        Path to GeoJSON file that outlines the region of interest (ROI). Only patches
        within the ROI(s) will be used.
    """

    def __init__(
        self,
        wsi_path: str | Path,
        patch_path: str | Path,
        um_px: float,
        patch_size: int,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        roi_path: str | Path | None = None,
    ):
        self.wsi_path = wsi_path
        self.patch_path = patch_path
        self.um_px = float(um_px)
        self.patch_size = int(patch_size)
        self.transform = transform
        self.roi_path = roi_path

        assert Path(wsi_path).exists(), "wsi path not found"
        assert Path(patch_path).exists(), "patch path not found"
        if roi_path is not None:
            assert Path(roi_path).exists(), "roi path not found"

        self.patches = _read_patch_coords(self.patch_path)

        # If an ROI is given, keep patches that intersect it.
        if self.roi_path is not None:
            self.patches = _filter_patches_in_rois(
                geojson_path=self.roi_path, coords=self.patches
            )
            if self.patches.shape[0] == 0:
                raise ValueError("No patches left after taking intersection with ROI")

        assert self.patches.ndim == 2, "expected 2D array of patch coordinates"
        # x, y, width, height
        assert self.patches.shape[1] == 4, "expected second dimension to have len 4"

    def worker_init(self, *_):
        self.slide = WSI(self.wsi_path)

    def __len__(self):
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
