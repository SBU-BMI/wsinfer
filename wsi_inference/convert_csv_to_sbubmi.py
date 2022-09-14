"""Convert CSV of model outputs to Stony Brook BMI formats.

This creates a JSON-lines file and a space-delimited table.
"""

from datetime import datetime
import json
from pathlib import Path
import time
import typing

import click
import large_image
import pandas as pd


PathType = typing.Union[str, Path]


def _box_to_polygon(
    *, minx: float, miny: float, width: float, height: float
) -> typing.List[typing.Tuple[float, float]]:
    """Get coordinates of a box polygon."""
    maxx = minx + width
    maxy = miny + height
    return [(maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny), (maxx, miny)]


def _get_timestamp() -> str:
    dt = datetime.now().astimezone()
    # 2022-08-29 13:46:28 EDT
    return dt.strftime("%Y-%m-%d_%H:%M:%S %Z")


def _row_to_heatmap_row(
    row: pd.Series,
    *,
    class_name: str,
    slide_width: int,
    slide_height: int,
    case_id: typing.Optional[str] = None,
    subject_id: typing.Optional[str] = None,
) -> typing.Dict:
    minx, miny, width, height = row["minx"], row["miny"], row["width"], row["height"]
    patch_area_base_pixels = width * height
    # Scale the values to be a ratio of the whole slide dimensions. All of the values
    # in the dictionary (except 'footprint') use these normalized coordinates.
    minx = float(minx / slide_width)
    miny = float(miny / slide_height)
    width = float(width / slide_width)
    height = float(height / slide_height)
    maxx = minx + width
    maxy = miny + height
    centerx = (minx + maxx) / 2
    centery = (miny + maxy) / 2
    coords = _box_to_polygon(minx=minx, miny=miny, width=width, height=height)

    # Get the probabilites from the model.
    if class_name not in row.index:
        raise KeyError(f"class name not found in results: {class_name}")
    class_probability: float = row[class_name]
    return {
        "type": "Feature",
        "parent_id": "self",
        "object_type": "heatmap_multiple",
        "x": centerx,
        "y": centery,
        "normalized": "true",
        "footprint": patch_area_base_pixels,
        "geometry": {
            "coordinates": [coords],
            "type": "Polygon",
        },
        "provenance": {
            "analysis": {
                "source": "computer",
                "execution_id": "tcga-brca-jakub-refactor-pipeline",
                "cancer_type": "quip",
                "study_id": "TCGA-BRCA",
                "computation": "heatmap",
            },
            "image": {
                "case_id": case_id,
                "subject_id": subject_id,
            },
        },
        "bbox": [minx, miny, maxx, maxy],
        "properties": {
            "multiheat_param": {
                "human_weight": -1,
                "metric_array": [class_probability, 0.0],
                "heatname_array": ["tumor", "necrosis"],
                "weight_array": ["0.5", "0.5"],
            },
            "metric_value": class_probability,
            "metric_type": "tile_dice",
            "human_mark": -1,
        },
        "date": {"$date": int(time.time())},
    }


def write_heatmap_json_like(
    input: PathType,
    output: PathType,
    class_name: str,
    slide_width: int,
    slide_height: int,
    case_id: typing.Optional[str] = None,
    subject_id: typing.Optional[str] = None,
) -> None:
    df = pd.read_csv(input)
    features = df.apply(
        _row_to_heatmap_row,
        axis=1,
        class_name=class_name,
        slide_width=slide_width,
        slide_height=slide_height,
        case_id=case_id,
        subject_id=subject_id,
    )
    features = features.tolist()
    features_json = (json.dumps(row) for row in features)
    with open(output, "w") as f:
        f.writelines(line + "\n" for line in features_json)


def write_heatmap_txt(input: PathType, output: PathType, class_name: str):
    def _apply_fn(row: pd.Series) -> str:
        minx, miny, w, h = (row["minx"], row["miny"], row["width"], row["height"])
        class_probability: float = row[class_name]
        centerx = round(minx + (w / 2))
        centery = round(miny + (h / 2))
        return f"{centerx} {centery} {class_probability}"

    df = pd.read_csv(input)
    # Each line is "centerx centery probability" and coordinates are in base pixels.
    lines: typing.List[str] = df.apply(_apply_fn, axis=1).tolist()
    with open(output, "w") as f:
        f.writelines(line + "\n" for line in lines)


def _version() -> str:
    """Closure to return version (avoid possibility of a circular import)."""
    from . import __version__

    return __version__


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output-jsonl",
    required=True,
    type=click.Path(dir_okay=False, exists=False, writable=True, path_type=Path),
    help="Path to write the JSON file (.json) file.",
)
@click.option(
    "--output-table",
    required=True,
    type=click.Path(dir_okay=False, exists=False, writable=True, path_type=Path),
    help="Path to write the text (.txt) file.",
)
@click.option(
    "--class-name",
    required=True,
    help="Name of the class to use for probability values.",
)
@click.option(
    "--slide",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to original whole slide image (mutually exclusive with slide_width"
    " and slide_height)",
)
@click.option(
    "--slide-width",
    type=int,
    default=None,
    help="Width of the slide in pixels (highest mag) (mutually exclusive with slide)",
)
@click.option(
    "--slide-height",
    type=int,
    default=None,
    help="Height of the slide in pixels (highest mag) (mutually exclusive with slide)",
)
@click.option("--subject-id", default=None, help="Subject ID")
@click.option("--case-id", default=None, help="Subject's case ID")
@click.version_option(version=_version())
def cli(
    *,
    input: Path,
    output_jsonl: Path,
    output_table: Path,
    class_name: str,
    slide: typing.Optional[Path] = None,
    slide_width: typing.Optional[int] = None,
    slide_height: typing.Optional[int] = None,
    subject_id: typing.Optional[str] = None,
    case_id: typing.Optional[str] = None,
):
    """Convert CSV of patch predictions to .txt and .json formats for use with Stony
    Brook Biomedical Informatics viewers.

    INPUT           Path to input CSV
    """
    click.echo(f"Reading CSV: {input}")
    if slide is None and (slide_width is None or slide_height is None):
        raise click.ClickException("slide or slide_width/slide_height must be provided")
    if slide is not None and (slide_width is not None or slide_height is not None):
        raise click.ClickException(
            "slide is mutually exclusive with slide_width and slide_height"
        )
    if slide_width is None and slide_height is None:
        ts: large_image.tilesource.TileSource = large_image.getTileSource(slide)
        slide_width = ts.getMetadata()["sizeX"]
        slide_height = ts.getMetadata()["sizeY"]
    print(f"Slide dimensions: {slide_width} x {slide_height} (width x height)")

    write_heatmap_json_like(
        input=input,
        output=output_jsonl,
        class_name=class_name,
        slide_width=slide_width,  # type: ignore
        slide_height=slide_height,  # type: ignore
        case_id=case_id,
        subject_id=subject_id,
    )
    click.secho(f"Saved JSON lines output to {output_jsonl}", fg="green")
    write_heatmap_txt(input=input, output=output_table, class_name=class_name)
    click.secho(f"Saved table output to {output_table}", fg="green")
