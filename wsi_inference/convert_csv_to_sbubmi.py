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
import random
import subprocess
import time
import typing

import click
import large_image
import multiprocessing
import numpy as np
import pandas as pd
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


def _get_git_info():
    def get_stdout(args):
        proc = subprocess.run(args, capture_output=True)
        return proc.stdout.decode().strip()

    git_remote = get_stdout("git config --get remote.origin.url".split())
    git_branch = get_stdout("git rev-parse --abbrev-ref HEAD".split())
    git_commit = get_stdout("git rev-parse HEAD".split())
    return {
        "git_remote": git_remote,
        "git_branch": git_branch,
        "git_commit": git_commit,
    }


def write_heatmap_and_meta_json_lines(
    input: PathType,
    output_heatmap: PathType,
    output_meta: PathType,
    slide_width: PathType,
    slide_height: PathType,
    execution_id: str,
    study_id: str,
    case_id: str,
    subject_id: str,
    class_name: str,
) -> None:
    """Write JSON-lines files for one slide."""

    # Run this before defining the function so the entire JSON file has the same
    # execution time value.
    execution_time = _get_timestamp()
    date = int(time.time())
    # TODO: Does not include model info: model_path, model_hash, model_url, model_ver.
    version_dict = _get_git_info()

    def row_to_json(row: pd.Series):
        minx, miny, width, height = (
            row["minx"],
            row["miny"],
            row["width"],
            row["height"],
        )
        patch_area_base_pixels = width * height
        # Scale the values to be a ratio of the whole slide dimensions. All of the
        # values in the dictionary (except 'footprint') use these normalized
        # coordinates.
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
        class_name_no_prob = class_name[5:]  # Remove "prob_" prefix.
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
                    "execution_time": execution_time,
                },
                "image": {
                    "case_id": case_id,
                    "subject_id": subject_id,
                },
                "version": version_dict,
            },
            "bbox": [minx, miny, maxx, maxy],
            "properties": {
                "multiheat_param": {
                    "human_weight": -1,
                    "metric_array": [class_probability],
                    "heatname_array": [class_name_no_prob],
                    "weight_array": ["1"],
                },
                "metric_value": class_probability,
                "metric_type": "tile_dice",
                "human_mark": -1,
            },
            "date": {"$date": date},
        }

    # Write heatmap JSON lines file.
    df = pd.read_csv(input)
    features = df.apply(row_to_json, axis=1)
    features = features.tolist()
    features = (json.dumps(row) for row in features)
    with open(output_heatmap, "w") as f:
        f.writelines(line + "\n" for line in features)

    # Write meta file.
    meta_dict = {
        "color": "yellow",  # This is copied from the lung cancer detection code.
        "title": execution_id,
        "image": {
            "case_id": case_id,
            "subject_id": subject_id,
        },
        "provenance": {
            "analysis_execution_id": execution_id,
            "analysis_execution_date": execution_time,
            "study_id": study_id,
            "type": "computer",
            "version": version_dict,
        },
        "submit_date": {"$date": date},
        "randval": random.uniform(0, 1),
    }
    with open(output_meta, "w") as f:
        json.dump(meta_dict, f)


def write_heatmap_txt(input: PathType, output: PathType):
    df = pd.read_csv(input)
    df.loc[:, "x_loc"] = (df.minx + (df.width / 2)).round().astype(int)
    df.loc[:, "y_loc"] = (df.miny + (df.height / 2)).round().astype(int)
    cols = [col for col in df.columns if col.startswith("prob_")]
    cols = [col[5:] for col in cols]  # remove 'prob_' prefix.
    cols = ["x_loc", "y_loc", *cols]
    df = df.loc[:, cols]
    df.to_csv(output, index=False, sep=" ")
    # TODO: should we write one file per label as well?


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
            output_heatmap = (
                output / "heatmap_json" / class_name / f"heatmap_{slide_id}.json"
            )
            output_meta = output_heatmap.parent / f"meta_{slide_id}.json"
            output_heatmap.parent.mkdir(parents=True, exist_ok=True)

            click.echo(f"Writing JSON lines file: {output_heatmap}")
            write_heatmap_and_meta_json_lines(
                input=input_csv,
                output_heatmap=output_heatmap,
                output_meta=output_meta,
                slide_width=ts.sizeX,
                slide_height=ts.sizeY,
                execution_id=execution_id,
                study_id=study_id,
                case_id=slide_id,  # TODO: should case_id be different?
                subject_id=slide_id,
                class_name=class_name,
            )

            output_txt_prediction = (
                output / "heatmap_txt" / class_name / f"prediction-{slide_id}"
            )
            write_heatmap_txt(input=input_csv, output=output_txt_prediction)

        # Do this once per WSI.
        if make_color_text:
            output_color = output / "heatmap_txt" / f"color-{slide_id}"
            write_color_txt(
                input=input_csv, output=output_color, ts=ts, num_processes=num_processes
            )

    click.secho("Finished.", fg="green")
