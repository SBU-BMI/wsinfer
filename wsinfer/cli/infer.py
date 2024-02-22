"""Detect cancerous regions in a whole slide image."""

from __future__ import annotations

import dataclasses
import getpass
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import wsinfer_zoo
import wsinfer_zoo.client
from wsinfer_zoo.client import HFModel
from wsinfer_zoo.client import ModelConfiguration

from ..modellib import models
from ..modellib.run_inference import run_inference
from ..patchlib import segment_and_patch_directory_of_slides
from ..qupath import make_qupath_project
from ..write_geojson import write_geojsons


def _num_cpus() -> int:
    """Get number of CPUs on the system."""
    try:
        return len(os.sched_getaffinity(0))
    # os.sched_getaffinity seems to be linux only.
    except AttributeError:
        count = os.cpu_count()  # potentially None
        return count or 0


def _inside_container() -> str:
    if Path("/.dockerenv").exists():
        return "yes, docker"
    elif (
        Path("/singularity").exists()
        or Path("/singularity.d").exists()
        or Path("/.singularity.d").exists()
    ):
        # TODO: apptainer might change the name of this directory.
        return "yes, apptainer/singularity"
    return "no"


def _get_timestamp() -> str:
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
    print("https://github.com/SBU-BMI/wsinfer/issues/new")
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
        click.secho(f"  Using {torch.cuda.device_count()} GPU(s)", fg="green")
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
            click.secho("  CPU-only version of PyTorch is installed", fg="yellow")
        click.secho("*******************************************", fg="yellow")
    elif not cuda_is_available:
        click.secho("\n*******************************************", fg="yellow")
        click.secho("GPU WILL NOT BE USED", fg="yellow")
        if torch.version.cuda is None:
            click.secho("  CUDA DEVICES NOT AVAILABLE", fg="yellow")
        click.secho("*******************************************", fg="yellow")


def _get_info_for_save(
    model_obj: models.LocalModelTorchScript | HFModel,
) -> dict[str, Any]:
    """Get dictionary with information about the run. To save as JSON in output dir."""

    import torch

    from .. import __version__

    here = Path(__file__).parent.resolve()

    def get_git_info() -> dict[str, str | bool]:
        here = Path(__file__).parent.resolve()

        def get_stdout(args: list[str]) -> str:
            proc = subprocess.run(args, capture_output=True, cwd=here)
            return "" if proc.returncode != 0 else proc.stdout.decode().strip()

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

    # Test if we are in a git repo. If we are, then get git info.
    git_program = shutil.which("git")
    git_installed = git_program is not None
    is_git_repo = False
    if git_installed:
        cmd = subprocess.run(
            [str(git_program), "branch"],
            cwd=here,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        is_git_repo = cmd.returncode == 0
    git_info = None
    if git_installed and is_git_repo:
        git_info = get_git_info()

    hf_info = None
    if hasattr(model_obj, "hf_info"):
        hf_info = dataclasses.asdict(model_obj.hf_info)

    return {
        "model": {
            "config": dataclasses.asdict(model_obj.config),
            "huggingface_location": hf_info,
            "path": str(model_obj.model_path),
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
            "wsinfer_zoo_version": wsinfer_zoo.__version__,
        },
        "timestamp": _get_timestamp(),
    }


@click.command()
@click.pass_context
@click.option(
    "-i",
    "--wsi-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing whole slide images. This directory can *only* contain"
    " whole slide images.",
)
@click.option(
    "-o",
    "--results-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory to store results. If directory exists, will skip"
    " whole slides for which outputs exist.",
)
@click.option(
    "-m",
    "--model",
    "model_name",
    type=click.Choice(sorted(wsinfer_zoo.client.load_registry().models.keys())),
    help="Name of the model to use from WSInfer Model Zoo. Mutually exclusive with"
    " --config.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Path to configuration for the trained model. Use this option if the"
        " model weights are not registered in wsinfer. Mutually exclusive with"
        "--model"
    ),
)
@click.option(
    "-p",
    "--model-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Path to the pretrained model. Use only when --config is passed. Mutually "
        "exclusive with --model."
    ),
)
@click.option(
    "-b",
    "--batch-size",
    type=click.IntRange(min=1),
    default=32,
    show_default=True,
    help="Batch size during model inference. If using multiple GPUs, increase the"
    " batch size.",
)
@click.option(
    "-n",
    "--num-workers",
    default=min(_num_cpus(), 8),  # Use at most 8 workers by default.
    show_default=True,
    type=click.IntRange(min=0),
    help="Number of workers to use for data loading during model inference (n=0 for"
    " single thread). Set this to the number of cores on your machine or lower.",
)
@click.option(
    "--speedup/--no-speedup",
    default=False,
    show_default=True,
    help="JIT-compile the model and apply inference optimizations. This imposes a"
    " startup cost but may improve performance overall.",
)
@click.option(
    "--qupath",
    is_flag=True,
    default=False,
    show_default=True,
    help="Create a QuPath project containing the inference results",
)
# Options for segmentation.
@click.option(
    "--seg-thumbsize",
    default=(2048, 2048),
    type=(int, int),
    help="The size of the slide thumbnail (in pixels) used for tissue segmentation."
    " The aspect ratio is preserved, and the longest side will have length"
    " max(thumbsize).",
)
@click.option(
    "--seg-median-filter-size",
    default=7,
    type=click.IntRange(min=3),
    help="The kernel size for median filtering. Must be greater than 1 and odd.",
)
@click.option(
    "--seg-binary-threshold",
    default=7,
    type=click.IntRange(min=1),
    help="The threshold for image binarization.",
)
@click.option(
    "--seg-closing-kernel-size",
    default=6,
    type=click.IntRange(min=1),
    help="The kernel size for binary closing (morphological operation).",
)
@click.option(
    "--seg-min-object-size-um2",
    default=200**2,
    type=click.FloatRange(min=0),
    help="The minimum size of an object to keep during tissue detection. If a"
    " contiguous object is smaller than this area, it replaced with background."
    " The default is 200um x 200um. The units of this argument are microns squared.",
)
@click.option(
    "--seg-min-hole-size-um2",
    default=190**2,
    type=click.FloatRange(min=0),
    help="The minimum size of a hole to keep as a hole. If a hole is smaller than this"
    " area, it is filled with foreground. The default is 190um x 190um. The units of"
    " this argument are microns squared.",
)
def run(
    ctx: click.Context,
    *,
    wsi_dir: Path,
    results_dir: Path,
    model_name: str | None,
    config: Path | None,
    model_path: Path | None,
    batch_size: int,
    num_workers: int = 0,
    speedup: bool = False,
    qupath: bool = False,
    seg_thumbsize: tuple[int, int],
    seg_median_filter_size: int,
    seg_binary_threshold: int,
    seg_closing_kernel_size: int,
    seg_min_object_size_um2: float,
    seg_min_hole_size_um2: float,
) -> None:
    """Run model inference on a directory of whole slide images.

    This command will create a tissue mask of each WSI. Then patch coordinates will be
    computed. The chosen model will be applied to each patch, and the results will be
    saved to a CSV in `RESULTS_DIR/model-output`.

    Example:

    CUDA_VISIBLE_DEVICES=0 wsinfer run --wsi-dir slides/ --results-dir results
    --model breast-tumor-resnet34.tcga-brca --batch-size 32 --num-workers 4

    To list all available models and weights, use `wsinfer-zoo ls`.
    """

    if model_name is None and config is None and model_path is None:
        raise click.UsageError(
            "one of --model or (--config and --model-path) is required."
        )
    elif (config is not None or model_path is not None) and model_name is not None:
        raise click.UsageError(
            "--config and --model-path are mutually exclusive with --model."
        )
    elif (config is not None) ^ (model_path is not None):  # XOR
        raise click.UsageError(
            "--config and --model-path must both be set if one is set."
        )

    if not wsi_dir.exists():
        raise FileNotFoundError(f"Whole slide image directory not found: {wsi_dir}")

    # Test that wsi dir actually includes files. This is here for an interesting edge
    # case. When using a Linux container and if the data directory is symlinked from a
    # different directory, both directories need to be bind mounted onto the container.
    # If only the symlinked directory is included, then the patching script will fail,
    # even though it looks like there are files in the wsi_dir directory.
    files_in_wsi_dir = [p for p in wsi_dir.iterdir() if p.is_file()]
    if not files_in_wsi_dir:
        raise FileNotFoundError(f"no files exist in the slide directory: {wsi_dir}")

    _print_system_info()

    print("\nCommand line arguments")
    print("----------------------")
    for key, value in ctx.params.items():
        print(f"{key} = {value}")
    print("----------------------\n")

    # Get weights object before running the patching script because we need to get the
    # necessary spacing and patch size.
    model_obj: HFModel | models.LocalModelTorchScript
    if model_name is not None:
        model_obj = models.get_registered_model(name=model_name)
    elif config is not None:
        with open(config) as f:
            _config_dict = json.load(f)
        model_config = ModelConfiguration.from_dict(_config_dict)
        model_obj = models.LocalModelTorchScript(
            config=model_config, model_path=str(model_path)
        )
        del _config_dict, model_config
    else:
        raise click.ClickException("Neither of --config and --model was passed")

    click.secho("\nFinding patch coordinates...\n", fg="green")

    segment_and_patch_directory_of_slides(
        wsi_dir=wsi_dir,
        save_dir=results_dir,
        patch_size_px=model_obj.config.patch_size_pixels,
        patch_spacing_um_px=model_obj.config.spacing_um_px,
        thumbsize=seg_thumbsize,
        median_filter_size=seg_median_filter_size,
        binary_threshold=seg_binary_threshold,
        closing_kernel_size=seg_closing_kernel_size,
        min_object_size_um2=seg_min_object_size_um2,
        min_hole_size_um2=seg_min_hole_size_um2,
    )

    if not results_dir.joinpath("patches").exists():
        raise click.ClickException(
            "No patches were created. Please see the logs above and check for errors."
            " It is possible that no tissue was detected in the slides. If that is the"
            " case, please try to use different --seg-* parameters, which will change"
            " how the segmentation is done. For example, a lower binary threshold may"
            " be set."
        )

    click.secho("\nRunning model inference.\n", fg="green")
    failed_patching, failed_inference = run_inference(
        wsi_dir=wsi_dir,
        results_dir=results_dir,
        model_info=model_obj,
        batch_size=batch_size,
        num_workers=num_workers,
        speedup=speedup,
    )

    if failed_patching:
        click.secho(f"\nPatching failed for {len(failed_patching)} slides", fg="yellow")
        click.secho("\n".join(failed_patching), fg="yellow")
    if failed_inference:
        click.secho(
            f"\nInference failed for {len(failed_inference)} slides", fg="yellow"
        )
        click.secho("\n".join(failed_inference), fg="yellow")

    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    run_metadata_outpath = results_dir / f"run_metadata_{timestamp}.json"
    click.echo(f"Saving metadata about run to {run_metadata_outpath}")
    run_metadata = _get_info_for_save(model_obj)
    with open(run_metadata_outpath, "w") as f:
        json.dump(run_metadata, f, indent=2)

    click.secho("Finished.", fg="green")

    csvs = list((results_dir / "model-outputs-csv").glob("*.csv"))
    write_geojsons(csvs, results_dir, num_workers)
    if qupath:
        make_qupath_project(wsi_dir, results_dir)
