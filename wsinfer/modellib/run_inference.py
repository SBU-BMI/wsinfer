"""Run inference.

From the original paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7369575/):
> In the prediction (test) phase, no data augmentation was applied except for the
> normalization of the color channels.
"""

import typing
from pathlib import Path
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch
import tqdm
import wsinfer_zoo.client

from .. import errors
from .data import WholeSlideImagePatches
from .models import LocalModel
from .models import get_pretrained_torch_module
from .models import jit_compile
from .transforms import make_compose_from_transform_config

PathType = Union[str, Path]


def run_inference(
    wsi_dir: PathType,
    results_dir: PathType,
    model_info: Union[wsinfer_zoo.client.HFModel, LocalModel],
    batch_size: int = 32,
    num_workers: int = 0,
    speedup: bool = False,
) -> Tuple[List[str], List[str]]:
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
    model_info :
        Instance of Weights including the model object and information about how to
        apply the model to new data.
    batch_size : int
        The batch size during the forward pass (default is 32).
    num_workers : int
        Number of workers for data loading (default is 0, meaning use a single thread).
    speedup : bool
        If True, JIT-compile the model. This has a startup cost but model inference
        should be faster (default False).

    Returns
    -------
    A tuple of two lists of strings. The first list contains the slide IDs for which
    patching failed, and the second list contains the slide IDs for which model
    inference failed.
    """
    # Make sure required directories exist.
    wsi_dir = Path(wsi_dir)
    if not wsi_dir.exists():
        raise errors.WholeSlideImageDirectoryNotFound(f"directory not found: {wsi_dir}")
    wsi_paths = list(wsi_dir.glob("*"))
    if not wsi_paths:
        raise errors.WholeSlideImagesNotFound(wsi_dir)
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise errors.ResultsDirectoryNotFound(results_dir)

    # Check patches directory.
    patch_dir = results_dir / "patches"
    if not patch_dir.exists():
        raise errors.PatchDirectoryNotFound("Results dir must include 'patches' dir")
    # Create the patch paths based on the whole slide image paths. In effect, only
    # create patch paths if the whole slide image patch exists.
    patch_paths = [patch_dir / p.with_suffix(".h5").name for p in wsi_paths]

    model_output_dir = results_dir / "model-outputs"
    model_output_dir.mkdir(exist_ok=True)

    model = get_pretrained_torch_module(model=model_info)
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

    transform = make_compose_from_transform_config(model_info.config.transform)

    failed_patching = [p.stem for p in patch_paths if not p.exists()]
    failed_inference: List[str] = []

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

        try:
            dset = WholeSlideImagePatches(
                wsi_path=wsi_path,
                patch_path=patch_path,
                um_px=model_info.config.spacing_um_px,
                transform=transform,
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
        slide_coords: List[np.ndarray] = []
        slide_probs: List[np.ndarray] = []
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
                # FIXME: should we include the slide path in the CSV? Probably not.
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
        prob_colnames = [f"prob_{c}" for c in model_info.config.class_names]
        slide_df.loc[:, prob_colnames] = slide_probs_arr
        slide_df.to_csv(slide_csv, index=False)
        print("-" * 40)

    return failed_patching, failed_inference
