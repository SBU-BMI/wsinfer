"""Run inference.

From the original paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7369575/):
> In the prediction (test) phase, no data augmentation was applied except for the
> normalization of the color channels.
"""

import argparse
import pathlib
import typing

import h5py
import large_image
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import tqdm

import models

PathType = typing.Union[str, pathlib.Path]


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
                "Could not find required key 'patch_level' in hdf5 of patch coordinates."
                " Has the version of CLAM been updated?"
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
        # TODO: consider whether we would ever want the alpha channel.
        patch_im = patch_im.convert("RGB")
        if self.transform is not None:
            patch_im = self.transform(patch_im)
        return patch_im, torch.as_tensor([minx, miny, width, height])


def run_inference_on_slides(
    *,
    wsi_paths: typing.Sequence[PathType],
    patch_paths: typing.Sequence[PathType],
    results_dir: PathType,
    um_px: float,
    model: torch.nn.Module,
    patch_size: int,
    batch_size: int = 64,
    num_workers: int = 0,
    disable_progbar: bool = False,
    classes: typing.Optional[typing.Sequence[str]] = None,
) -> None:
    """"""

    results_dir = pathlib.Path(results_dir)
    model_output_dir = results_dir / "model-outputs"
    model_output_dir.mkdir(exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(
                (patch_size, patch_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            # This is valid for
            transforms.Normalize(
                mean=[0.7238, 0.5716, 0.6779],
                std=[0.1120, 0.1459, 0.1089],
            ),
        ]
    )
    # results_for_all_slides: typing.List[pd.DataFrame] = []
    for wsi_path, patch_path in zip(wsi_paths, patch_paths):
        print("----")
        print(f"Slide path: {wsi_path}")
        print(f"Patch path: {patch_path}")

        slide_csv_name = pathlib.Path(wsi_path).with_suffix(".csv").name
        slide_csv = model_output_dir / slide_csv_name
        if slide_csv.exists():
            print("Output CSV exists... skipping.")
            print(slide_csv)
            continue

        dset = WholeSlideImagePatches(
            wsi_path=wsi_path,
            patch_path=patch_path,
            um_px=um_px,
            transform=transform,
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
        for batch_imgs, batch_coords in tqdm.tqdm(loader, disable=disable_progbar):
            assert batch_imgs.shape[0] == batch_coords.shape[0], "length mismatch"
            with torch.no_grad():
                logits: torch.Tensor = model(batch_imgs.to(device)).detach().cpu()
            # probs has shape (batch_size, num_classes)
            probs = torch.nn.functional.softmax(logits, dim=1)

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
        num_classes = slide_probs_arr.shape[1]
        class_names = classes or [f"cls{i}" for i in range(num_classes)]
        slide_df.loc[:, class_names] = slide_probs_arr
        slide_df.to_csv(slide_csv, index=False)
    return


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wsi_dir", required=True, help="Path to input whole slide image.")
    # TODO: add save_dir
    p.add_argument(
        "--results_dir",
        required=True,
        help="Path to directory with patch results.",
    )
    p.add_argument(
        "--patch_size",
        type=int,
        required=True,
        help=(
            "Patch size for input to model in pixels at the desired spacing. The"
            " same spacing is used as when patching was done."
        ),
    )
    p.add_argument(
        "--um_px",
        type=float,
        required=True,
        help="Scaling for patches (in micrometer per pixel).",
    )
    p.add_argument("--model", type=str, required=True, help="Name of the model")
    p.add_argument("--num_classes", type=int, required=True, help="Number of classes.")
    p.add_argument("--weights", type=str, help="Path to state dict weights for model.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--classes", nargs="+", help="Names of the classes (in order)")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--disable_progbar", action="store_true")

    args = p.parse_args()

    args.wsi_dir = pathlib.Path(args.wsi_dir)
    if not args.wsi_dir.exists():
        raise FileNotFoundError(args.wsi_dir)
    wsi_paths = list(args.wsi_dir.glob("*"))
    if not wsi_paths:
        raise FileNotFoundError(f"no files found in {args.wsi_dir}")

    args.results_dir = pathlib.Path(args.results_dir)
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {args.results_dir}")

    if args.classes is not None:
        if len(args.classes) != args.num_classes:
            raise ValueError("length of --classes must be equal to --num_classes")

    # Check patches directory.
    patch_dir = args.results_dir / "patches"
    if not patch_dir.exists():
        raise FileNotFoundError("Results dir must include 'patches' dir")
    patch_paths = [patch_dir / p.with_suffix(".h5").name for p in wsi_paths]
    patch_paths_notfound = [p for p in patch_paths if not p.exists()]
    if patch_paths_notfound:
        raise FileNotFoundError(
            f"could not find patch hdf5 files: {patch_paths_notfound}"
        )

    print(patch_paths)

    model = models.create_model(
        args.model, num_classes=args.num_classes, state_dict_path=args.weights
    )

    run_inference_on_slides(
        wsi_paths=wsi_paths,
        patch_paths=patch_paths,
        results_dir=args.results_dir,
        um_px=args.um_px,
        model=model,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        disable_progbar=args.disable_progbar,
        classes=args.classes,
    )
