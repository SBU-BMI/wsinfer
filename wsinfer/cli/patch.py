from typing import Optional

import click

from .._patchlib.create_patches_fp import create_patches as _create_patches


@click.command()
@click.option(
    "--source",
    required=True,
    type=click.Path(exists=True),
    help="patch to directory containing whole slide image files",
)
@click.option("--step-size", default=None, type=int, help="Step size for patching.")
@click.option("--patch-size", required=True, type=int, help="Patch size in pixels.")
@click.option("--auto-skip/--no-auto-skip", default=True, help="Skip existing outputs.")
@click.option(
    "--save-dir",
    required=True,
    type=click.Path(),
    help="Directory to save processed data",
)
@click.option(
    "--preset",
    default="tcga.csv",
    help="Predefined profile of default segmentation and filter parameters (.csv)",
)
@click.option(
    "--process-list", help="Name of list of images to process with parameters (.csv)"
)
@click.option(
    "--patch-spacing",
    required=True,
    type=float,
    help="Patch spacing in micrometers per pixel.",
)
@click.option(
    "--segmentation-dir",
    type=click.Path(),
    help="Directory containing .pkl files with tissue segmentations",
)
def patch(
    source: str,
    step_size: Optional[int],
    patch_size: int,
    auto_skip: bool,
    save_dir: str,
    preset: Optional[str],
    process_list: Optional[str],
    patch_spacing: float,
    segmentation_dir: Optional[str],
):
    """Patchify a directory of whole slide images."""

    print("create_patches_fp.py  Copyright (C) 2022  Mahmood Lab")
    print("This program comes with ABSOLUTELY NO WARRANTY.")
    print("This is free software, and you are welcome to redistribute it")
    print("under certain conditions.")

    _create_patches(
        source=source,
        step_size=step_size,
        patch_size=patch_size,
        patch_spacing=patch_spacing,
        save_dir=save_dir,
        patch=True,
        seg=True,
        stitch=True,
        auto_skip=auto_skip,
        preset=preset,
        process_list=process_list,
        segmentation_dir=segmentation_dir,
    )
