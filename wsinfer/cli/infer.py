"""Detect cancerous regions in a whole slide image."""

import getpass
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import typing

import click

from ..modellib.run_inference import run_inference
from ..modellib import models
from ..patchlib.create_dense_patch_grid import create_grid_and_save_multi_slides
from ..patchlib.create_patches_fp import create_patches

PathType = typing.Union[str, pathlib.Path]


def _inside_container() -> str:
    if pathlib.Path("/.dockerenv").exists():
        return "yes, docker"
    elif (
        pathlib.Path("/singularity").exists()
        or pathlib.Path("/singularity.d").exists()
        or pathlib.Path("/.singularity.d").exists()
    ):
        # TODO: apptainer might change the name of this directory.
        return "yes, apptainer/singularity"
    return "no"


def _get_timestamp() -> str:
    from datetime import datetime

    dt = datetime.now().astimezone()
    # Thu Aug 25 23:32:17 2022 EDT
    return dt.strftime("%c %Z")


def _print_system_info() -> None:
    """Print information about the system."""
    import torch
    import torchvision
    from .. import __version__

    click.secho(f"\nRunning wsinfer version {__version__}", fg="green")
    print("\nIf you run into issues, please submit a new issue at")
    print("https://github.com/kaczmarj/patch-classification-pipeline/issues/new")
    print("\nSystem information")
    print("------------------")
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
    cuda_is_available = torch.cuda.is_available()
    if cuda_is_available:
        click.secho("GPU available", fg="green")
        cuda_ver = torch.version.cuda or "NOT FOUND"
        print(f"  CUDA version: {cuda_ver}")
    else:
        click.secho("GPU not available", bg="red", fg="black")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    if torch.version.cuda is None:
        click.secho("\n*******************************************", fg="yellow")
        click.secho("GPU WILL NOT BE USED", fg="yellow")
        if torch.version.cuda is None:
            click.secho("  CPU-only version of PyTorch", fg="yellow")
        click.secho("*******************************************", fg="yellow")
    elif not torch.cuda.is_available():
        click.secho("\n*******************************************", fg="yellow")
        click.secho("GPU WILL NOT BE USED", fg="yellow")
        if torch.version.cuda is None:
            click.secho("  CUDA DEVICES NOT AVAILABLE", fg="yellow")
        click.secho("*******************************************", fg="yellow")


def _sha256sum(path: PathType) -> str:
    """Calculate SHA256 of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(1024 * 64)  # 64 kb
            if not data:
                break
            sha.update(data)
    return sha.hexdigest()


def _get_info_for_save(weights: models.Weights):
    """Get dictionary with information about the run. To save as JSON in output dir."""

    import torch
    from .. import __version__

    here = pathlib.Path(__file__).parent.resolve()

    def get_git_info():
        here = pathlib.Path(__file__).parent.resolve()

        def get_stdout(args) -> str:
            proc = subprocess.run(args, capture_output=True, cwd=here)
            return proc.stdout.decode().strip()

        git_remote = get_stdout("git config --get remote.origin.url".split())
        git_branch = get_stdout("git rev-parse --abbrev-ref HEAD".split())
        git_commit = get_stdout("git rev-parse HEAD".split())

        # https://stackoverflow.com/a/3879077/5666087
        cmd = subprocess.run("git diff-index --quiet HEAD --".split(), cwd=here)
        uncommitted_changes = cmd.returncode != 0
        return {
            "git_remote": git_remote,
            "git_branch": git_branch,
            "git_commit": git_commit,
            "uncommitted_changes": uncommitted_changes,
        }

    weights_path = pathlib.Path(torch.hub.get_dir()) / "checkpoints" / weights.file_name
    shasum = _sha256sum(weights_path) if weights_path.exists() else None

    # Test if we are in a git repo. If we are, then get git info.
    cmd = subprocess.run(
        "git branch".split(),
        cwd=here,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if cmd.returncode == 0:
        git_info = get_git_info()
    else:
        git_info = None
    del cmd, here  # For sanity.

    return {
        "model": {
            "architecture": "resnet34",
            "weights_url": weights.url,
            "weights_path": str(weights_path) if weights_path.exists() else None,
            "weights_sha256": shasum,
            "class_names": weights.class_names,
            "num_classes": weights.num_classes,
            "metadata": weights.metadata or None,
        },
        "input_data": {
            "patch_size_pixels": weights.patch_size_pixels,
            "spacing_um_px": weights.spacing_um_px,
        },
        "runtime": {
            "version": __version__,
            "working_dir": os.getcwd(),
            "args": " ".join(sys.argv),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "in_container": _inside_container(),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "git": git_info,
        },
        "timestamp": _get_timestamp(),
    }


@click.command(context_settings=dict(auto_envvar_prefix="WSINFER"))
@click.pass_context
@click.option(
    "--wsi_dir",
    type=click.Path(
        exists=True, file_okay=False, path_type=pathlib.Path, resolve_path=True
    ),
    required=True,
    help="Directory containing whole slide images. This directory can *only* contain"
    " whole slide images.",
)
@click.option(
    "--results_dir",
    type=click.Path(file_okay=False, path_type=pathlib.Path, resolve_path=True),
    required=True,
    help="Directory to store results. If directory exists, will skip"
    " whole slides for which outputs exist.",
)
@click.option(
    "--model",
    type=click.Choice(models.list_models()),
    required=True,
    help="Model architecture to use.",
)
@click.option(
    "--weights",
    type=str,
    required=True,
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
@click.option(
    "--dense-grid/--no-dense-grid",
    default=False,
    show_default=True,
    help="Use a dense grid of patch coordinates. Patches will be present even if no"
    " tissue is present",
)
def cli(
    ctx: click.Context,
    *,
    wsi_dir: pathlib.Path,
    results_dir: pathlib.Path,
    model: str,
    weights: str,
    batch_size: int,
    num_workers: int = 0,
    dense_grid: bool = False,
):
    """Run model inference on a directory of whole slide images.

    This command will create a tissue mask of each WSI. Then patch coordinates will be
    computed. The chosen model will be applied to each patch, and the results will be
    saved to a CSV in `RESULTS_DIR/model-output`.

    Example:

    CUDA_VISIBLE_DEVICES=0 wsinfer run --wsi_dir slides/ --results_dir results
    --model resnet34 --weights TCGA-BRCA-v1 --batch_size 32 --num_workers 4
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

    _print_system_info()

    print("\nCommand line arguments")
    print("----------------------")
    for key, value in ctx.params.items():
        print(f"{key} = {value}")
    print("----------------------\n")

    # Get model before running the patching script because we need to get the necessary
    # spacing and patch size.
    weights_obj = models.create_model(model, weights=weights)

    click.secho("\nFinding patch coordinates...\n", fg="green")
    if dense_grid:
        click.echo("Not using a tissue mask.")
        create_grid_and_save_multi_slides(
            wsi_dir=wsi_dir,
            results_dir=results_dir,
            orig_patch_size=weights_obj.patch_size_pixels,
            patch_spacing_um_px=weights_obj.spacing_um_px,
        )
    else:
        create_patches(
            source=str(wsi_dir),
            save_dir=str(results_dir),
            patch_size=weights_obj.patch_size_pixels,
            patch_spacing=weights_obj.spacing_um_px,
            seg=True,
            patch=True,
            preset="tcga.csv",
        )

    click.secho("\nRunning model inference.\n", fg="green")
    run_inference(
        wsi_dir=wsi_dir,
        results_dir=results_dir,
        weights=weights_obj,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    run_metadata_outpath = results_dir / "run_metadata.json"
    click.echo(f"Saving metadata about run to {run_metadata_outpath}")
    run_metadata = _get_info_for_save(weights_obj)
    with open(run_metadata_outpath, "w") as f:
        json.dump(run_metadata, f, indent=2)

    click.secho("Finished.", fg="green")
