"""Convert CSV of model outputs to Stony Brook BMI formats.

Directories:
heatmap_jsons (same as json)
heatmap_txt
json (same as heatmap_jsons)
patch-level-color
patch-level-lym
patch-level-merged
"""

from datetime import datetime
import json
from pathlib import Path
import time
import typing

import click
import large_image
import multiprocessing
import numpy as np
import pandas as pd
import shutil
import tqdm


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


def write_color_txt(
    input: PathType,
    output: PathType,
    ts: large_image.tilesource.TileSource,
    num_processes: int = 6,
):
    def whiteness(arr):
        arr = np.asarray(arr)
        return np.std(arr, axis=(0, 1)).mean()

    def blackness(arr):
        arr = np.asarray(arr)
        return arr.mean()

    def redness(arr):
        arr = np.asarray(arr)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        # boolean multiplication is logical and
        return np.mean((r >= 190) * (g <= 100) * (b <= 100))

    global get_color  # Hack to please multiprocessing.

    def get_color(row: pd.Series):
        arr, _ = ts.getRegion(
            format=large_image.constants.TILE_FORMAT_NUMPY,
            region=dict(
                left=row["minx"],
                top=row["miny"],
                width=row["width"],
                height=row["height"],
            ),
        )
        white = whiteness(arr)
        black = blackness(arr)
        red = redness(arr)
        return white, black, red

    df = pd.read_csv(input)
    df_rows_as_dicts = df.to_dict("records")

    print("Calculating color information for each patch in the whole slide image...")
    print("This might take some time.")
    print("Use a large --num-processes value to hopefully speed it up.")

    # https://stackoverflow.com/a/45276885/5666087
    with multiprocessing.Pool(num_processes) as p:
        results = list(
            tqdm.tqdm(
                p.imap(get_color, df_rows_as_dicts, chunksize=16),  # type: ignore
                total=len(df_rows_as_dicts),
            )
        )
    df.loc[:, ["whiteness", "blackness", "redness"]] = results
    cx = df.minx + ((df.minx + df.width) / 2)
    assert np.array_equal(cx.astype(int), cx), "not all center x's are integer"
    df.loc[:, "cx"] = cx.astype(int)
    del cx  # sanity

    cy = df.miny + ((df.miny + df.height) / 2)
    assert np.array_equal(cy.astype(int), cy), "not all center y's are integer"
    df.loc[:, "cy"] = cy.astype(int)
    del cy

    df = df.loc[:, ["cx", "cy", "whiteness", "blackness", "redness"]]
    df.to_csv(output, header=False, index=False, sep=" ")


def _version() -> str:
    """Return version (avoid possibility of a circular import)."""
    from . import __version__

    return __version__


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Directory in which to save outputs.",
)
# TODO: how are multi-class predictions saved?
@click.option(
    "--class-name",
    required=True,
    help="Name of the class to use for probability values.",
)
@click.option(
    "--slide",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to original whole slide image (mutually exclusive with slide_width"
    " and slide_height)",
)
@click.option("--subject-id", default=None, help="Subject ID")
@click.option("--case-id", default=None, help="Subject's case ID")
@click.option(
    "--num-processes",
    type=int,
    default=4,
    help="Number of processes to use when creating color text file.",
)
@click.version_option(version=_version())
def cli(
    *,
    input: Path,
    output_dir: Path,
    class_name: str,
    slide: Path,
    subject_id: typing.Optional[str] = None,
    case_id: typing.Optional[str] = None,
    num_processes: int = 4,
):
    """Convert CSV of patch predictions to .txt and .json formats for use with Stony
    Brook Biomedical Informatics viewers.

    INPUT           Path to input CSV
    """
    click.echo(f"Reading CSV: {input}")
    ts: large_image.tilesource.TileSource = large_image.getTileSource(slide)
    slide_width: int = ts.getMetadata()["sizeX"]
    slide_height: int = ts.getMetadata()["sizeY"]
    print(f"Slide dimensions: {slide_width} x {slide_height} (width x height)")

    slide_id = slide.stem

    if not output_dir.exists():
        print(f"Making directory {output_dir}")
        output_dir.mkdir()

    # Set up directories.
    (output_dir / "json").mkdir(exist_ok=True)
    (output_dir / "heatmap_jsons").mkdir(exist_ok=True)
    (output_dir / "heatmap_txt").mkdir(exist_ok=True)
    (output_dir / "patch-level-color").mkdir(exist_ok=True)
    (output_dir / "patch-level-merged").mkdir(exist_ok=True)

    # Write JSON files with heatmap info.
    jsonl_output = output_dir / "json" / f"heatmap_{slide_id}.json"
    write_heatmap_json_like(
        input=input,
        output=jsonl_output,
        class_name=class_name,
        slide_width=slide_width,
        slide_height=slide_height,
        case_id=case_id,
        subject_id=subject_id,
    )
    shutil.copy(jsonl_output, output_dir / "heatmap_jsons")
    click.secho("Saved JSON heatmaps.", fg="green")

    # Write tables with predictions.
    table_output = output_dir / "heatmap_txt" / f"prediction-{slide_id}"
    write_heatmap_txt(input=input, output=table_output, class_name=class_name)
    shutil.copy(table_output, output_dir / "patch-level-merged")
    click.secho("Saved tables with predictions.", fg="green")

    # TODO: add patch-level-CLASS files.

    # Write color tables.
    color_output = output_dir / "heatmap_txt" / f"prediction-{slide_id}"
    write_color_txt(
        input=input, output=color_output, ts=ts, num_processes=num_processes
    )
    shutil.copy(color_output, output_dir / "patch-level-color")
    click.secho("Saved table with color info.", fg="green")

    click.secho("Finished.", fg="green")
