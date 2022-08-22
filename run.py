"""Detect cancerous regions in a whole slide image."""

import argparse
import os
import pathlib
import subprocess
import sys
import typing

PathType = typing.Union[str, pathlib.Path]

_script_path = pathlib.Path(__file__).resolve().parent


def run_patching(
    slides_dir: PathType, save_dir: PathType, patch_size: int, patch_spacing: float
) -> subprocess.CompletedProcess:
    args = [
        sys.executable,
        "create_patches_fp.py",
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
    cwd = _script_path / "CLAM"
    print("Running the patching script:")
    print(f"CWD {cwd}")
    print(" ".join(args), flush=True)
    proc = subprocess.run(
        args, stdout=sys.stdout, stderr=sys.stderr, cwd=cwd, check=True
    )
    return proc


def run_inference(
    slides_dir: PathType,
    save_dir: PathType,
    patch_size: int,
    patch_spacing: float,
    model: str,
    num_classes: int,
    weights: PathType,
    batch_size: int,
    classes: typing.Optional[typing.Sequence[str]] = None,
    num_workers: int = 0,
) -> subprocess.CompletedProcess:
    save_dir = pathlib.Path(save_dir)
    patch_dir = save_dir / "patches"
    if not patch_dir.exists():
        raise FileNotFoundError(f"Patch directory not found: {patch_dir}")

    if not pathlib.Path(weights).exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    args: typing.List[str] = [
        sys.executable,
        "run_inference.py",
        "--wsi_dir",
        str(slides_dir),
        "--results_dir",
        str(save_dir),
        "--patch_size",
        str(patch_size),
        "--um_px",
        str(patch_spacing),
        "--model",
        model,
        "--num_classes",
        str(num_classes),
        "--weights",
        str(weights),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
    ]
    if classes is not None:
        args.extend(["--classes", *classes])
    cwd = _script_path
    print("Running the model inference script:")
    print(f"CWD {cwd}")
    print(" ".join(args), flush=True)
    proc = subprocess.run(
        args, stdout=sys.stdout, stderr=sys.stderr, cwd=cwd, check=True
    )
    return proc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--wsi_dir",
        required=True,
        help="Path to directory containing (only) whole slide images.",
    )
    p.add_argument(
        "--results_dir", required=True, help="Path in which to save results."
    )
    p.add_argument(
        "--patch_size",
        type=int,
        required=True,
        help="Patch size for input to model in pixels at the desired spacing.",
    )
    p.add_argument(
        "--um_px",
        type=float,
        required=True,
        help="Scaling for patches (in micrometer per pixel).",
    )
    p.add_argument("--model", required=True, help="Name of the model")
    p.add_argument("--num_classes", type=int, required=True, help="Number of classes.")
    p.add_argument(
        "--weights", required=True, help="Path to state dict weights for model."
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--classes", nargs="+", help="Names of the classes (in order)")
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    # The patching command runs in a different working directory, so let's make the
    # paths absolute.
    # TODO: consider how to handle existing files.
    args.wsi_dir = pathlib.Path(args.wsi_dir).resolve()
    args.results_dir = pathlib.Path(args.results_dir).resolve()
    args.weights = pathlib.Path(args.weights).resolve()

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is None or cuda_visible_devices == "":
        print("*********************************************************")
        print("NO GPU WILL BE USED BECAUSE CUDA_VISIBLE_DEVICES IS EMPTY")
        print("*********************************************************")

    if not args.wsi_dir.exists():
        raise FileNotFoundError(
            f"Whole slide image directory not found: {args.wsi_dir}"
        )

    # Test that wsi dir actually includes files.
    files_in_wsi_dir = [p for p in args.wsi_dir.glob("*") if p.exists()]
    if not files_in_wsi_dir:
        raise FileNotFoundError(
            f"no files exist in the slide directory: {args.wsi_dir}"
        )

    print("\nArguments")
    print("---------")
    for key, value in vars(args).items():
        print(f"{key} = {value}")
    print("---------\n")

    run_patching(
        slides_dir=args.wsi_dir,
        save_dir=args.results_dir,
        patch_size=args.patch_size,
        patch_spacing=args.um_px,
    )

    run_inference(
        slides_dir=args.wsi_dir,
        save_dir=args.results_dir,
        patch_size=args.patch_size,
        patch_spacing=args.um_px,
        model=args.model,
        num_classes=args.num_classes,
        weights=args.weights,
        batch_size=args.batch_size,
        classes=args.classes,
        num_workers=args.num_workers,
    )

    return


if __name__ == "__main__":
    main()
