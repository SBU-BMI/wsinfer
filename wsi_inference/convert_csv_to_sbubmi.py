"""Convert CSV of model outputs to Stony Brook BMI formats.

This creates a JSON-lines file and a space-delimited table.
"""

from datetime import datetime
import json
from pathlib import Path
import time
import typing

import click
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
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


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
    # Scale the values to be a ratio of the whole slide dimensions.
    minx = float(minx / slide_width)
    miny = float(miny / slide_height)
    width = float(width / slide_width)
    height = float(height / slide_height)
    maxx = minx + width
    maxy = miny + height
    coords = _box_to_polygon(minx=minx, miny=miny, width=width, height=height)

    # Get the probabilites from the model.
    if class_name not in row.index:
        raise KeyError(f"class name not found in results: {class_name}")
    class_probability: float = row[class_name]
    d = {
        "type": "Feature",
        "parent_id": "self",
        # This seems to be the only un-normalized value.
        "footprint": patch_area_base_pixels,
        "x": (minx + maxx) / 2,
        "y": (miny + maxy) / 2,
        "normalized": "true",
        "object_type": "heatmap_multiple",
        "bbox": [
            minx / slide_width,
            miny / slide_height,
            maxx / slide_width,
            maxy / slide_height,
        ],
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
        "properties": {
            # TODO: what should metric_* be?
            "metric_value": 0.0,
            "metric_type": "tile_dice",
            "human_mark": -1,
            "multiheat_param": {
                "human_weight": -1,
                "weight_array": ["1.0"],
                "heatname_array": [class_name],
                "metric_array": [class_probability],
            },
        },
        "provenance": {
            "image": {
                "case_id": case_id,
                "subject_id": subject_id,
            },
            "analysis": {
                "study_id": "brca",
                "execution_id": "cancer-brca-high_res",
                "source": "computer",
                "computation": "heatmap",
                "execution_date": _get_timestamp(),
            },
            "version": {},
        },
        # Epoch time in seconds.
        "date": {"$date": int(time.time())},
    }
    return d


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
@click.argument(
    "output_jsonl",
    type=click.Path(dir_okay=False, exists=False, writable=True, path_type=Path),
)
@click.argument(
    "output_table",
    type=click.Path(dir_okay=False, exists=False, writable=True, path_type=Path),
)
@click.option(
    "--class-name",
    required=True,
    help="Name of the class to use for probability values.",
)
@click.option(
    "--slide-width",
    required=True,
    type=int,
    help="Width of the slide in pixels (highest mag)",
)
@click.option(
    "--slide-height",
    required=True,
    type=int,
    help="Height of the slide in pixels (highest mag)",
)
@click.option("--subject-id", default=None, help="Subject ID", show_default=True)
@click.option("--case-id", default=None, help="Subject's case ID", show_default=True)
@click.version_option(version=_version())
def cli(
    *,
    input: Path,
    output_jsonl: Path,
    output_table: Path,
    class_name: str,
    slide_width: int,
    slide_height: int,
    subject_id: typing.Optional[str],
    case_id: typing.Optional[str],
):
    """Convert CSV of patch predictions to a GeoJSON file.

    GeoJSON files can be used with pathology viewers like QuPath.

    INPUT           Path to input CSV

    OUTPUT_JSONL    Path to output JSON lines file (with .jsonl extension)

    OUTPUT_TABLE    Path to output table files (with .txt extension)
    """
    click.echo(f"Reading CSV: {input}")

    write_heatmap_json_like(
        input=input,
        output=output_jsonl,
        class_name=class_name,
        slide_width=slide_width,
        slide_height=slide_height,
        case_id=case_id,
        subject_id=subject_id,
    )
    click.secho(f"Saved JSON lines output to {output_jsonl}", fg="green")
    write_heatmap_txt(input=input, output=output_table, class_name=class_name)
    click.secho(f"Saved table output to {output_table}", fg="green")
