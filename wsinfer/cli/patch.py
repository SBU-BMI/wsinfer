from __future__ import annotations

from pathlib import Path

import click

from ..patchlib import segment_and_patch_directory_of_slides


@click.command()
@click.option(
    "-i",
    "--wsi-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path, resolve_path=True),
    required=True,
    help="Directory containing whole slide images. This directory can *only* contain"
    " whole slide images.",
)
@click.option(
    "-o",
    "--results-dir",
    type=click.Path(file_okay=False, path_type=Path, resolve_path=True),
    required=True,
    help="Directory to store patch results. If directory exists, will skip"
    " whole slides for which outputs exist.",
)
@click.option("--patch-size-px", required=True, type=int, help="Patch size in pixels.")
@click.option(
    "--patch-spacing-um-px",
    required=True,
    type=float,
    help="Physical spacing of the patch in micrometers per pixel.",
)
@click.option(
    "--thumbsize",
    default=(2048, 2048),
    type=(int, int),
    help="The size of the slide thumbnail (in pixels) used for tissue segmentation."
    " The aspect ratio is preserved, and the longest side will have length"
    " max(thumbsize).",
)
@click.option(
    "--median-filter-size",
    default=7,
    type=click.IntRange(min=3),
    help="The kernel size for median filtering. Must be greater than 1 and odd.",
)
@click.option(
    "--binary-threshold",
    default=7,
    type=click.IntRange(min=1),
    help="The threshold for image binarization.",
)
@click.option(
    "--closing-kernel-size",
    default=6,
    type=click.IntRange(min=1),
    help="The kernel size for binary closing (morphological operation).",
)
@click.option(
    "--min-object-size-um2",
    default=200**2,
    type=click.FloatRange(min=0),
    help="The minimum size of an object to keep during tissue detection. If a"
    " contiguous object is smaller than this area, it replaced with background."
    " The default is 200um x 200um. The units of this argument are microns squared.",
)
@click.option(
    "--min-hole-size-um2",
    default=190**2,
    type=click.FloatRange(min=0),
    help="The minimum size of a hole to keep as a hole. If a hole is smaller than this"
    " area, it is filled with foreground. The default is 190um x 190um. The units of"
    " this argument are microns squared.",
)
def patch(
    wsi_dir: str,
    results_dir: str,
    patch_size_px: int,
    patch_spacing_um_px: float,
    thumbsize: tuple[int, int],
    median_filter_size: int,
    binary_threshold: int,
    closing_kernel_size: int,
    min_object_size_um2: float,
    min_hole_size_um2: float,
):
    """Patch a directory of whole slide iamges."""
    segment_and_patch_directory_of_slides(
        wsi_dir=wsi_dir,
        save_dir=results_dir,
        patch_size_px=patch_size_px,
        patch_spacing_um_px=patch_spacing_um_px,
        thumbsize=thumbsize,
        median_filter_size=median_filter_size,
        binary_threshold=binary_threshold,
        closing_kernel_size=closing_kernel_size,
        min_object_size_um2=min_object_size_um2,
        min_hole_size_um2=min_hole_size_um2,
    )
