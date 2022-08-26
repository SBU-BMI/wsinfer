"""Detect cancerous regions in a whole slide image."""

import getpass
import os
import pathlib
import platform
import subprocess
import sys
import typing

import click

from .modellib.run_inference import run_inference
from .modellib.models import list_models

PathType = typing.Union[str, pathlib.Path]


def run_patching(
    slides_dir: PathType, save_dir: PathType, patch_size: int, patch_spacing: float
) -> subprocess.CompletedProcess:
    args = [
        # This command is created when installing this package.
        "wsi_create_patches",
        "--source",
        str(slides_dir),
        "--save_dir",
        str(save_dir),
        "--patch_size",
        str(patch_size),
        "--patch_spacing",
        str(patch_spacing),
        "--seg",
        "--patch",
        "--stitch",
        "--preset",
        # Consider customizing this...
        "tcga.csv",
    ]
    click.secho("\nRunning the patching script.\n", fg="green")
    print(" ".join(args), "\n", flush=True)
    proc = subprocess.run(args, stdout=sys.stdout, stderr=sys.stderr, check=True)
    return proc


def _inside_container() -> str:
    if pathlib.Path("/.dockerenv").exists():
        return "yes, docker"
    elif pathlib.Path("/singularity.d").exists():
        # TODO: apptainer might change the name of this directory.
        return "yes, apptainer/singularity"
    return "no"


def _get_timestamp() -> str:
    from datetime import datetime

    dt = datetime.now().astimezone()
    # Thu Aug 25 23:32:17 2022 EDT
    return dt.strftime("%c %Z")


def _print_info() -> None:
    """Print information about the system."""
    import torch
    import torchvision
    from . import __version__

    click.secho(f"\nRunning wsi_inference version {__version__}", fg="green")
    print("\nIf you run into issues, please submit a new issue at")
    print("https://github.com/kaczmarj/patch-classification-pipeline/issues/new")
    print("\nInformation")
    print("-----------")
    print(f"Timestamp: {_get_timestamp()}")
    print(f"{platform.platform()}")
    try:
        print(f"User: {getpass.getuser()}")
    except KeyError:
        # If /etc/passwd does not include the username of the current user ID, a
        # KeyError is thrown. This could happen in a Docker image running as a different
        # user with `--user $(id -u):$(id -g)` but that does not bind mount the
        # /etc/passwd file.
        print("User: UNKNOWN")
    print(f"Hostname: {platform.node()}")
    print(f"Working directory: {os.getcwd()}")
    print(f"In container: {_inside_container()}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"  Torch version: {torch.__version__}")
    print(f"  Torchvision version: {torchvision.__version__}")
    cuda_ver = torch.version.cuda or "NOT FOUND"
    print(f"  CUDA version: {cuda_ver}")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")

    if torch.version.cuda is None:
        click.secho("\n*******************************************", fg="yellow")
        click.secho("GPU WILL NOT BE USED", fg="yellow")
        if torch.version.cuda is None:
            click.secho("  CPU-only version of PyTorch is being used", fg="yellow")
        click.secho("*******************************************", fg="yellow")
    elif cuda_visible_devices == "NOT SET" or cuda_visible_devices == "":
        click.secho("\n*******************************************", fg="yellow")
        click.secho("GPU WILL NOT BE USED", fg="yellow")
        if torch.version.cuda is None:
            click.secho("  CUDA_VISIBLE_DEVICES empty or not set", fg="yellow")
        click.secho("*******************************************", fg="yellow")


@click.command()
@click.pass_context
@click.option(
    "--wsi_dir",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    required=True,
    help="Directory containing whole slide images. This directory can *only* contain"
    " whole slide images.",
)
@click.option(
    "--results_dir",
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    required=True,
    help="Directory to store results. If directory exists, will skip"
    " whole slides for which outputs exist.",
)
@click.option(
    "--patch_size",
    type=click.IntRange(min=1),
    required=True,
    help="Size of square patch.",
)
@click.option(
    "--um_px",
    type=click.FloatRange(min=0.0),
    required=True,
    help="Spacing for patches (micrometers per pixel)",
)
@click.option(
    "--model",
    type=click.Choice(list_models()),
    required=True,
    help="Model architecture to use.",
)
@click.option(
    "--num_classes",
    type=click.IntRange(min=1),
    required=True,
    help="The number of classes the model outputs.",
)
@click.option(
    "--weights",
    type=str,
    default="TCGA-BRCA-v1",
    show_default=True,
    help="Weights to use for the model.",
)
@click.option(
    "--batch_size",
    type=click.IntRange(min=1),
    default=32,
    show_default=True,
    help="Batch size during model inference.",
)
@click.option(
    "--num_workers",
    default=0,
    show_default=True,
    help="Number of workers to use for data loading during model inference (default=0"
    " for single thread). A reasonable value is 8.",
)
@click.version_option()
def cli(
    ctx: click.Context,
    *,
    wsi_dir: pathlib.Path,
    results_dir: pathlib.Path,
    patch_size: int,
    um_px: float,
    model: str,
    num_classes: int,
    weights: str,
    batch_size: int,
    num_workers: int = 0,
):
    """Run model inference on a directory of whole slide images (WSI).

    This command will create a tissue mask of each WSI. Then patch coordinates will be
    computed. The chosen model will be applied to each patch, and the results will be
    saved to a CSV in `RESULTS_DIR/model-output`.

    Example:

    CUDA_VISIBLE_DEVICES=0 wsi_run --wsi_dir slides/ --results_dir results
    --patch_size 350 --um_px 0.250 --model resnet34
    --weights weights/resnet34.pt --num_classes 2 --batch_size 32 --num_workers 4
    """

    wsi_dir = wsi_dir.resolve()
    results_dir = results_dir.resolve()

    if not wsi_dir.exists():
        raise FileNotFoundError(f"Whole slide image directory not found: {wsi_dir}")

    # Test that wsi dir actually includes files. This is here for an interesting edge
    # case. When using a Linux container and if the data directory is symlinked from a
    # different directory, both directories need to be bind mounted onto the container.
    # If only the symlinked directory is included, then the patching script will fail,
    # even though it looks like there are files in the wsi_dir directory.
    files_in_wsi_dir = [p for p in wsi_dir.glob("*") if p.exists()]
    if not files_in_wsi_dir:
        raise FileNotFoundError(f"no files exist in the slide directory: {wsi_dir}")

    if weights != "TCGA-BRCA-v1":
        raise ctx.fail("Only 'TCGA-BRCA-v1' weights are available at this time.")

    _print_info()

    print("\nArguments")
    print("---------")
    for key, value in ctx.params.items():
        print(f"{key} = {value}")
    print("---------")

    run_patching(
        slides_dir=wsi_dir,
        save_dir=results_dir,
        patch_size=patch_size,
        patch_spacing=um_px,
    )

    click.secho("\nRunning model inference.\n", fg="green")
    run_inference(
        wsi_dir=wsi_dir,
        results_dir=results_dir,
        patch_size=patch_size,
        um_px=um_px,
        model_name=model,
        num_classes=num_classes,
        weights=weights,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    click.secho("Finished.", fg="green")

    return
