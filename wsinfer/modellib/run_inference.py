"""Run inference.

From the original paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7369575/):
> In the prediction (test) phase, no data augmentation was applied except for the
> normalization of the color channels.
"""

import pathlib
import typing
import warnings

import h5py
import large_image
import numpy as np
import pandas as pd
from PIL import Image

# Torch and torchvision are not in the requirements (setup.cfg) because the installation
# can vary based on platform and hardware.
try:
    import torch
except ImportError:
    raise ImportError(
        "Please install torch and torchvision: https://pytorch.org/get-started/locally/"
    )
import tqdm

from . import models

PathType = typing.Union[str, pathlib.Path]


class WholeSlideImageDirectoryNotFound(FileNotFoundError):
    ...


class WholeSlideImagesNotFound(FileNotFoundError):
    ...


class ResultsDirectoryNotFound(FileNotFoundError):
    ...


class PatchDirectoryNotFound(FileNotFoundError):
    ...


class PatchFilesNotFoundWarning(UserWarning):
    ...


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
        Scale of the resulting patches. Use 0.5 for 20x magnification.
    transform : callable, optional
        A callable to modify a retrieved patch. The callable must accept a
        PIL.Image.Image instance and return a torch.Tensor.
    """

    def __init__(
        self,
        wsi_path: PathType,
        patch_path: PathType,
        um_px: float,
        transform: typing.Optional[typing.Callable[[Image.Image], torch.Tensor]] = None,
    ):
        self.wsi_path = wsi_path
        self.patch_path = patch_path
        self.um_px = float(um_px)
        self.transform = transform

        assert pathlib.Path(wsi_path).exists(), "wsi path not found"
        assert pathlib.Path(patch_path).exists(), "patch path not found"

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
        assert self.patches.ndim == 2, "expected 2D array of patch coordinates"
        # x, y, width, height
        assert self.patches.shape[1] == 4, "expected second dimension to have len 4"

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
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
        return patch_im, torch.as_tensor([minx, miny, width, height])


def run_inference(
    wsi_dir: PathType,
    results_dir: PathType,
    weights: models.Weights,
    batch_size: int = 32,
    num_workers: int = 0,
) -> None:
    """Run model inference on a directory of whole slide images and save results to CSV.

    This assumes the patching has already been done and the results are stored in
    `results_dir`. An error will be raised otherwise.

    Output CSV files are written to `{results_dir}/model-outputs/`.

    Parameters
    ----------
    wsi_dir : str or pathlib.Path
        Directory containing whole slide images. This directory can *only* contain
        whole slide images. Otherwise, an error will be raised during model inference.
    results_dir : str or pathlib.Path
        Directory containing results of patching.
    weights : wsinfer.modellib.models.Weights
        Instance of Weights including the model object and information about how to
        apply the model to new data.
    batch_size : int
        The batch size during the forward pass (default is 32).
    num_workers : int
        Number of workers for data loading (default is 0, meaning use a single thread).

    Returns
    -------
    None
    """
    # Make sure required directories exist.
    wsi_dir = pathlib.Path(wsi_dir)
    if not wsi_dir.exists():
        raise WholeSlideImageDirectoryNotFound(f"directory not found: {wsi_dir}")
    wsi_paths = list(wsi_dir.glob("*"))
    if not wsi_paths:
        raise WholeSlideImagesNotFound(wsi_dir)
    results_dir = pathlib.Path(results_dir)
    if not results_dir.exists():
        raise ResultsDirectoryNotFound(results_dir)

    # Check patches directory.
    patch_dir = results_dir / "patches"
    if not patch_dir.exists():
        raise PatchDirectoryNotFound("Results dir must include 'patches' dir")
    # Create the patch paths based on the whole slide image paths. In effect, only
    # create patch paths if the whole slide image patch exists.
    patch_paths = [patch_dir / p.with_suffix(".h5").name for p in wsi_paths]
    patch_paths_notfound = [p for p in patch_paths if not p.exists()]
    if patch_paths_notfound:
        warnings.warn(
            "Patch extraction seems to have failed for the following slides:"
            + " ".join(str(p.stem) for p in patch_paths_notfound),
            category=PatchFilesNotFoundWarning,
        )

    if weights.model is None:
        raise RuntimeError("model cannot be None in the weights object")

    model_output_dir = results_dir / "model-outputs"
    model_output_dir.mkdir(exist_ok=True)

    model = weights.model
    if model is None:
        raise ValueError("Model was not instantiated... use `create_model`.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # results_for_all_slides: typing.List[pd.DataFrame] = []
    for i, (wsi_path, patch_path) in enumerate(zip(wsi_paths, patch_paths)):
        print(f"Slide {i+1} of {len(wsi_paths)}")
        print(f" Slide path: {wsi_path}")
        print(f" Patch path: {patch_path}")

        slide_csv_name = pathlib.Path(wsi_path).with_suffix(".csv").name
        slide_csv = model_output_dir / slide_csv_name
        if slide_csv.exists():
            print("Output CSV exists... skipping.")
            print(slide_csv)
            continue

        if not patch_path.exists():
            print(f"Skipping because patch file not found: {patch_path}")
            continue

        dset = WholeSlideImagePatches(
            wsi_path=wsi_path,
            patch_path=patch_path,
            um_px=weights.spacing_um_px,
            transform=weights.transform,
        )

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

    return
