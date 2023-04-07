"""Run inference.

From the original paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7369575/):
> In the prediction (test) phase, no data augmentation was applied except for the
> normalization of the color channels.
"""

from pathlib import Path
import typing
import warnings

import h5py
import large_image
import numpy as np
import pandas as pd
from PIL import Image
import torch
import tqdm

from .models import Weights

PathType = typing.Union[str, Path]


class WholeSlideImageDirectoryNotFound(FileNotFoundError):
    ...


class WholeSlideImagesNotFound(FileNotFoundError):
    ...


class ResultsDirectoryNotFound(FileNotFoundError):
    ...


class PatchDirectoryNotFound(FileNotFoundError):
    ...


# Set the maximum number of TileSource objects to cache. We use 1 to minimize how many
# file handles we keep open.
large_image.config.setConfig("cache_tilesource_maximum", 1)


def _read_patch_coords(path: PathType) -> np.ndarray:
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
    *, geojson_path: PathType, coords: np.ndarray
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
    from shapely.geometry import box, shape

    with open(geojson_path) as f:
        geo = geojson.load(f)
    if not geo.is_valid:
        raise ValueError("GeoJSON of ROI is not valid")
    for roi in geo["features"]:
        assert roi.is_valid, "an ROI geometry is not valid"
    print("Converting geojson to shapely...")
    geoms_rois = [shape(roi["geometry"]) for roi in geo["features"]]
    coords = coords.copy()
    coords[:, 2] += coords[:, 0]  # Calculate maxx.
    coords[:, 3] += coords[:, 1]  # Calculate maxy.
    print("Making boxes...")
    boxes = [box(*coords[idx]) for idx in range(coords.shape[0])]
    tree = STRtree(boxes)
    print("Finding intersecting boxes...")
    _, intersecting_ids = tree.query(geoms_rois, predicate="intersects")
    intersecting_ids = np.sort(np.unique(intersecting_ids))
    return coords[intersecting_ids]


class WholeSlideImagePatches(torch.utils.data.Dataset):
    """Dataset of one whole slide image.

    This object retrieves patches from a whole slide image on the fly.

    Parameters
    ----------
    wsi_path : str, Path
        Path to whole slide image file.
    patch_path : str, Path
        Path to HDF5 file with coordinates of input image.
    um_px : float
        Scale of the resulting patches. Use 0.5 for 20x magnification.
    transform : callable, optional
        A callable to modify a retrieved patch. The callable must accept a
        PIL.Image.Image instance and return a torch.Tensor.
    roi_path : str, Path, optional
        Path to GeoJSON file that outlines the region of interest (ROI). Only patches
        within the ROI(s) will be used.
    """

    def __init__(
        self,
        wsi_path: PathType,
        patch_path: PathType,
        um_px: float,
        transform: typing.Optional[typing.Callable[[Image.Image], torch.Tensor]] = None,
        roi_path: typing.Optional[PathType] = None,
    ):
        self.wsi_path = wsi_path
        self.patch_path = patch_path
        self.um_px = float(um_px)
        self.transform = transform
        self.roi_path = roi_path

        assert Path(wsi_path).exists(), "wsi path not found"
        assert Path(patch_path).exists(), "patch path not found"
        if roi_path is not None:
            assert Path(roi_path).exists(), "roi path not found"

        self.tilesource: large_image.tilesource.TileSource = large_image.getTileSource(
            self.wsi_path
        )
        # Disable the tile cache. We wrap this in a try-except because we are accessing
        # a private attribute. It is possible that this attribute will change names
        # in the future, and if that happens, we do not want to raise errors.
        try:
            self.tilesource.cache._Cache__maxsize = 0
        except AttributeError:
            pass

        self.patches = _read_patch_coords(self.patch_path)

        # If an ROI is given, keep patches that intersect it.
        if self.roi_path is not None:
            print("Filtering patches...")
            self.patches = _filter_patches_in_rois(
                geojson_path=self.roi_path, coords=self.patches
            )
            if self.patches.shape[0] == 0:
                raise ValueError("No patches left after taking intersection with ROI")

        assert self.patches.ndim == 2, "expected 2D array of patch coordinates"
        # x, y, width, height
        assert self.patches.shape[1] == 4, "expected second dimension to have len 4"

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(
        self, idx: int
    ) -> typing.Tuple[typing.Union[Image.Image, torch.Tensor], torch.Tensor]:
        coords: typing.Sequence[int] = self.patches[idx]
        assert len(coords) == 4, "expected 4 coords (minx, miny, width, height)"
        minx, miny, width, height = coords
        source_region = dict(
            left=minx, top=miny, width=width, height=height, units="base_pixels"
        )
        target_scale = dict(mm_x=self.um_px / 1000)

        patch_im, _ = self.tilesource.getRegionAtAnotherScale(
            sourceRegion=source_region,
            targetScale=target_scale,
            format=large_image.tilesource.TILE_FORMAT_PIL,
        )
        patch_im = patch_im.convert("RGB")
        if self.transform is not None:
            patch_im = self.transform(patch_im)
        if not isinstance(patch_im, (Image.Image, torch.Tensor)):
            raise TypeError(
                f"patch image must be an Image of Tensor, but got {type(patch_im)}"
            )
        return patch_im, torch.as_tensor([minx, miny, width, height])


def jit_compile(
    model: torch.nn.Module,
) -> typing.Union[torch.jit.ScriptModule, torch.nn.Module, typing.Callable]:
    """JIT-compile a model for inference."""
    noncompiled = model
    device = next(model.parameters()).device
    # Attempt to script. If it fails, return the original.
    test_input = torch.ones(1, 3, 224, 224).to(device)
    w = "Warning: could not JIT compile the model. Using non-compiled model instead."
    # TODO: consider freezing the model as well.
    # PyTorch 2.x has torch.compile.
    if hasattr(torch, "compile"):
        # Try to get the most optimized model.
        try:
            return torch.compile(model, fullgraph=True, mode="max-autotune")
        except Exception:
            pass
        try:
            return torch.compile(model, mode="max-autotune")
        except Exception:
            pass
        try:
            return torch.compile(model)
        except Exception:
            warnings.warn(w)
            return noncompiled
    # For pytorch 1.x, use torch.jit.script.
    else:
        try:
            mjit = torch.jit.script(model)
            with torch.no_grad():
                mjit(test_input)
        except Exception:
            warnings.warn(w)
            return noncompiled
        # Now that we have scripted the model, try to optimize it further. If that
        # fails, return the scripted model.
        try:
            mjit_frozen = torch.jit.freeze(mjit)
            mjit_opt = torch.jit.optimize_for_inference(mjit_frozen)
            with torch.no_grad():
                mjit_opt(test_input)
            return mjit_opt
        except Exception:
            return mjit


def run_inference(
    wsi_dir: PathType,
    results_dir: PathType,
    weights: Weights,
    batch_size: int = 32,
    num_workers: int = 0,
    speedup: bool = False,
    roi_dir: typing.Optional[PathType] = None,
) -> typing.Tuple[typing.List[str], typing.List[str]]:
    """Run model inference on a directory of whole slide images and save results to CSV.

    This assumes the patching has already been done and the results are stored in
    `results_dir`. An error will be raised otherwise.

    Output CSV files are written to `{results_dir}/model-outputs/`.

    Parameters
    ----------
    wsi_dir : str or Path
        Directory containing whole slide images. This directory can *only* contain
        whole slide images. Otherwise, an error will be raised during model inference.
    results_dir : str or Path
        Directory containing results of patching.
    weights : wsinfer._modellib.models.Weights
        Instance of Weights including the model object and information about how to
        apply the model to new data.
    batch_size : int
        The batch size during the forward pass (default is 32).
    num_workers : int
        Number of workers for data loading (default is 0, meaning use a single thread).
    speedup : bool
        If True, JIT-compile the model. This has a startup cost but model inference
        should be faster (default False).
    roi_dir : str, Path, optional
        Directory containing GeoJSON files that outlines the regions of interest (ROI).
        Only patches within the ROI(s) will be used. The GeoJSON files must have the
        extension ".json".

    Returns
    -------
    A tuple of two lists of strings. The first list contains the slide IDs for which
    patching failed, and the second list contains the slide IDs for which model
    inference failed.
    """
    # Make sure required directories exist.
    wsi_dir = Path(wsi_dir)
    if not wsi_dir.exists():
        raise WholeSlideImageDirectoryNotFound(f"directory not found: {wsi_dir}")
    wsi_paths = list(wsi_dir.glob("*"))
    if not wsi_paths:
        raise WholeSlideImagesNotFound(wsi_dir)
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise ResultsDirectoryNotFound(results_dir)

    # Check patches directory.
    patch_dir = results_dir / "patches"
    if not patch_dir.exists():
        raise PatchDirectoryNotFound("Results dir must include 'patches' dir")
    # Create the patch paths based on the whole slide image paths. In effect, only
    # create patch paths if the whole slide image patch exists.
    patch_paths = [patch_dir / p.with_suffix(".h5").name for p in wsi_paths]

    model_output_dir = results_dir / "model-outputs"
    model_output_dir.mkdir(exist_ok=True)

    model = weights.load_model()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    if speedup:
        if typing.TYPE_CHECKING:
            model = typing.cast(torch.nn.Module, jit_compile(model))
        else:
            model = jit_compile(model)

    failed_patching = [p.stem for p in patch_paths if not p.exists()]
    failed_inference: typing.List[str] = []

    # Get paths to ROI geojson files.
    if roi_dir is not None:
        roi_paths = [Path(roi_dir) / p.with_suffix(".json").name for p in wsi_paths]
    else:
        roi_paths = None

    # results_for_all_slides: typing.List[pd.DataFrame] = []
    for i, (wsi_path, patch_path) in enumerate(zip(wsi_paths, patch_paths)):
        print(f"Slide {i+1} of {len(wsi_paths)}")
        print(f" Slide path: {wsi_path}")
        print(f" Patch path: {patch_path}")

        slide_csv_name = Path(wsi_path).with_suffix(".csv").name
        slide_csv = model_output_dir / slide_csv_name
        if slide_csv.exists():
            print("Output CSV exists... skipping.")
            print(slide_csv)
            continue

        if not patch_path.exists():
            print(f"Skipping because patch file not found: {patch_path}")
            continue

        roi_path = None
        if roi_paths is not None:
            roi_path = roi_paths[i]
            # We grab all potential names of ROI paths, but we do not require all of
            # them to exist. We only use those that exist.
            if not roi_path.exists():
                roi_path = None
            else:
                print(f" ROI path: {roi_path}")

        try:
            dset = WholeSlideImagePatches(
                wsi_path=wsi_path,
                patch_path=patch_path,
                um_px=weights.spacing_um_px,
                transform=weights.transform,
                roi_path=roi_path,
            )
        except Exception:
            failed_inference.append(wsi_dir.stem)
            continue

        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # Store the coordinates and model probabiltiies of each patch in this slide.
        # This lets us know where the probabiltiies map to in the slide.
        slide_coords: typing.List[np.ndarray] = []
        slide_probs: typing.List[np.ndarray] = []
        for batch_imgs, batch_coords in tqdm.tqdm(loader):
            assert batch_imgs.shape[0] == batch_coords.shape[0], "length mismatch"
            with torch.no_grad():
                logits: torch.Tensor = model(batch_imgs.to(device)).detach().cpu()
            # probs has shape (batch_size, num_classes) or (batch_size,)
            if len(logits.shape) > 1 and logits.shape[1] > 1:
                probs = torch.nn.functional.softmax(logits, dim=1)
            else:
                probs = torch.sigmoid(logits.squeeze(1))
            slide_coords.append(batch_coords.numpy())
            slide_probs.append(probs.numpy())

        slide_coords_arr = np.concatenate(slide_coords, axis=0)
        slide_df = pd.DataFrame(
            dict(
                slide=wsi_path,
                minx=slide_coords_arr[:, 0],
                miny=slide_coords_arr[:, 1],
                width=slide_coords_arr[:, 2],
                height=slide_coords_arr[:, 3],
            )
        )
        slide_probs_arr = np.concatenate(slide_probs, axis=0)
        # Use 'prob-' prefix for all classes. This should make it clearer that the
        # column has probabilities for the class. It also makes it easier for us to
        # identify columns associated with probabilities.
        prob_colnames = [f"prob_{c}" for c in weights.class_names]
        slide_df.loc[:, prob_colnames] = slide_probs_arr
        slide_df.to_csv(slide_csv, index=False)
        print("-" * 40)

    return failed_patching, failed_inference
