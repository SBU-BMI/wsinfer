"""Convert CSVs of model outputs to GeoJSON files.

GeoJSON files can be loaded into whole slide image viewers like QuPath.
"""

import json
from pathlib import Path
import typing

import click
import pandas as pd
import tqdm


def _box_to_polygon(
    *, minx: int, miny: int, width: int, height: int
) -> typing.List[typing.Tuple[int, int]]:
    """Get coordinates of a box polygon."""
    maxx = minx + width
    maxy = miny + height
    return [(maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny), (maxx, miny)]


def _row_to_geojson(row: pd.Series, prob_cols: typing.List[str]) -> typing.Dict:
    """Convert information about one tile to a single GeoJSON feature."""
    minx, miny, width, height = row["minx"], row["miny"], row["width"], row["height"]
    coords = _box_to_polygon(minx=minx, miny=miny, width=width, height=height)
    prob_dict = row[prob_cols].to_dict()
    measurements = [{"name": k, "value": v} for k, v in prob_dict.items()]
    return {
        "type": "Feature",
        "id": "PathTileObject",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
        "properties": {
            "isLocked": True,
            # measurements is a list of {"name": str, "value": float} dicts.
            # https://qupath.github.io/javadoc/docs/qupath/lib/measurements/MeasurementList.html
            "measurements": measurements,
            # classification is a dict of "name": str and optionally "color": int.
            # https://qupath.github.io/javadoc/docs/qupath/lib/objects/classes/PathClass.html
            # We do not include classification because we do not enforce a single class
            # per tile.
            # "classification": {"name": class_name},
        },
    }


def _dataframe_to_geojson(df: pd.DataFrame, prob_cols: typing.List[str]) -> typing.Dict:
    """Convert a dataframe of tiles to GeoJSON format."""
    features = df.apply(_row_to_geojson, axis=1, prob_cols=prob_cols)
    return {
        "type": "FeatureCollection",
        "features": features.tolist(),
    }


def convert(input, output) -> None:
    df = pd.read_csv(input)
    prob_cols = [col for col in df.columns.tolist() if col.startswith("prob_")]
    if not prob_cols:
        raise click.ClickException("Did not find any columns with prob_ prefix.")
    geojson = _dataframe_to_geojson(df, prob_cols)
    with open(output, "w") as f:
        json.dump(geojson, f)


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
def cli(*, results_dir: Path, output: Path):
    """Convert model outputs to GeoJSON format.

    GeoJSON files can be used with pathology viewers like QuPath.

    RESULTS_DIR     Path to results directory (containing model-outputs dir).

    OUTPUT          Path to output directory in which to save GeoJSON files.
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

    csvs = list((results_dir / "model-outputs").glob("*.csv"))
    if not csvs:
        raise click.ClickException("No CSVs found. Did you generate model outputs?")

    output.mkdir(exist_ok=False)

    for input_csv in tqdm.tqdm(csvs):
        output_path = output / input_csv.with_suffix(".json").name
        convert(input=input_csv, output=output_path)

    click.secho(f"Saved outputs to {output}", fg="green")
