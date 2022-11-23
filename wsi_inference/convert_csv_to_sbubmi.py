"""Convert CSV of model outputs to Stony Brook BMI formats.

Output directory tree:
├── heatmap_jsons
│   ├── heatmap-SLIDEID.json
│   └── meta-SLIDEID.json
└── heatmap_txt
    ├── color-SLIDEID
    └── prediction-SLIDEID
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


def _row_to_heatmap_json_row(
    row: pd.Series,
    *,
    class_name: str,
    slide_width: int,
    slide_height: int,
    execution_id: str,
    study_id: str,
    case_id: str,
    subject_id: str,
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
                "execution_id": execution_id,
                "cancer_type": "quip",
                "study_id": study_id,
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
                "metric_array": [class_probability],
                "heatname_array": ["tumor"],  # TODO: change the name.
                "weight_array": ["1"],
            },
            "metric_value": class_probability,
            "metric_type": "tile_dice",
            "human_mark": -1,
        },
        "date": {"$date": int(time.time())},
    }


def write_heatmap_json_lines(
    input: PathType,
    output: PathType,
    slide_width: PathType,
    slide_height: PathType,
    execution_id: str,
    study_id: str,
    case_id: str,
    subject_id: str,
) -> None:
    df = pd.read_csv(input)
    features = df.apply(
        _row_to_heatmap_json_row,
        axis=1,
        slide_width=slide_width,
        slide_height=slide_height,
        execution_id=execution_id,
        study_id=study_id,
        case_id=case_id,
        subject_id=subject_id,
    )
    features = features.tolist()
    features = (json.dumps(row) for row in features)
    with open(output, "w") as f:
        f.writelines(line + "\n" for line in features)


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
@click.argument(
    "results_dir",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=Path, resolve_path=True
    ),
)
@click.argument(
    "output",
    type=click.Path(exists=False, path_type=Path, resolve_path=True),
)
@click.option(
    "--wsi-dir",
    required=True,
    help="Directory with whole slide images.",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=Path, resolve_path=True
    ),
)
@click.option("--execution-id", required=True, help="Unique execution ID for this run.")
@click.option("--study-id", required=True, help="Study ID, like TCGA-BRCA.")
@click.option(
    "--make-color-text/--no-make-color-text",
    default=False,
    help="Make text files with color information for each patch. NOTE: this can add"
    " several minutes of processing time per slide.",
)
@click.option(
    "--num-processes",
    type=int,
    default=4,
    help="Number of processes to use when `--make-color-text` is enabled.",
)
@click.version_option(version=_version())
def cli(
    *,
    results_dir: Path,
    output: Path,
    wsi_dir: Path,
    execution_id: str,
    study_id: str,
    make_color_text: bool = False,
    num_processes: int = 4,
):
    """Convert CSV of patch predictions to .txt and .json formats for use with Stony
    Brook Biomedical Informatics viewers.

    RESULTS_DIR     Path to results directory (containing model-outputs dir).

    OUTPUT          Path to output directory in which to save files.
    """
    if not results_dir.exists():
        raise click.ClickException(f"results_dir does not exist: {results_dir}")
    if output.exists():
        raise click.ClickException("Output directory already exists.")
    if (
        not (results_dir / "model-outputs").exists()
        and (results_dir / "patches").exists()
    ):
        raise click.ClickException(
            "Model outputs have not been generated yet. Please run model inference."
        )
    if not (results_dir / "model-outputs").exists():
        raise click.ClickException(
            "Expected results_dir to contain a 'model-outputs' directory but it does"
            " not. Please provide the path to the directory that contains"
            " model-outputs, masks, and patches."
        )
    if not wsi_dir.exists():
        raise click.ClickException("Whole slide image directory does not exist.")

    csvs = list((results_dir / "model-outputs").glob("*.csv"))
    if not csvs:
        raise click.ClickException("No CSVs found. Did you generate model outputs?")

    output.mkdir(exist_ok=False)

    # Get the output classes of the model. We will create separate directories for each
    # label name. But by default, we do not include labels that start with "no", like
    # "notils" and "notumor".
    class_names: typing.Sequence[str] = pd.read_csv(csvs[0]).columns
    class_names = [n for n in class_names if n.startswith("prob_")]
    ignore_names = {"notils", "notumor"}
    class_names = [n for n in class_names if n not in ignore_names]

    for input_csv in tqdm.tqdm(csvs):
        slide_id = input_csv.stem
        wsi_file = wsi_dir / f"{slide_id}.svs"
        if not wsi_file.exists():
            click.secho(f"WSI file not found: {wsi_file}", bg="red")
            click.secho("Skipping...", bg="red")
            continue
        ts = large_image.getTileSource(wsi_file)
        if ts.sizeX is None or ts.sizeY is None:
            click.secho(f"Unknown size for WSI: {wsi_file}", bg="red")
            click.secho("Skipping...", bg="red")
            continue

        for class_name in class_names:
            output_path = (
                output / "heatmap_json" / class_name / f"heatmap_{slide_id}.json"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            click.echo(f"Writing JSON file: {output_path}")
            write_heatmap_json_lines(
                input=input_csv,
                output=output_path,
                slide_width=ts.sizeX,
                slide_height=ts.sizeY,
                execution_id=execution_id,
                study_id=study_id,
                case_id=slide_id,  # TODO: should case_id be different?
                subject_id=slide_id,
            )

            # TODO: write meta json files.
            # TODO: write text files.

    return

    # Write tables with predictions.
    table_output = output_dir / "heatmap_txt" / f"prediction-{slide_id}"
    write_heatmap_txt(input=input, output=table_output, class_name=class_name)
    shutil.copy(table_output, output_dir / "patch-level-merged")
    click.secho("Saved tables with predictions.", fg="green")

    # TODO: add patch-level-CLASS files.

    # Write color tables.
    if make_color_text:
        color_output = output_dir / "heatmap_txt" / f"prediction-{slide_id}"
        write_color_txt(
            input=input, output=color_output, ts=ts, num_processes=num_processes
        )
        shutil.copy(color_output, output_dir / "patch-level-color")
        click.secho("Saved table with color info.", fg="green")

    click.secho("Finished.", fg="green")
