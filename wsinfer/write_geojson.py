"""Convert CSVs of model outputs to GeoJSON files.

GeoJSON files can be loaded into whole slide image viewers like QuPath.
"""

from __future__ import annotations

import json
import shutil
import uuid
from functools import partial
from pathlib import Path

import click
import pandas as pd
from tqdm.contrib.concurrent import process_map


def _box_to_polygon(
    *, minx: int, miny: int, width: int, height: int
) -> list[tuple[int, int]]:
    """Get coordinates of a box polygon."""
    maxx = minx + width
    maxy = miny + height
    return [(maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny), (maxx, miny)]


def _row_to_geojson(row: pd.Series, prob_cols: list[str]) -> dict:
    """Convert information about one tile to a single GeoJSON feature."""
    minx, miny, width, height = row["minx"], row["miny"], row["width"], row["height"]
    coords = _box_to_polygon(minx=minx, miny=miny, width=width, height=height)
    prob_dict = row[prob_cols].to_dict()

    measurements = {}
    for k, v in prob_dict.items():
        measurements[k] = v

    return {
        "type": "Feature",
        "id": str(uuid.uuid4()),
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
        "properties": {
            "isLocked": True,
            # measurements is a list of {"name": str, "value": float} dicts.
            # https://qupath.github.io/javadoc/docs/qupath/lib/measurements/MeasurementList.html
            "measurements": measurements,
            "objectType": "tile"
            # classification is a dict of "name": str and optionally "color": int.
            # https://qupath.github.io/javadoc/docs/qupath/lib/objects/classes/PathClass.html
            # We do not include classification because we do not enforce a single class
            # per tile.
            # "classification": {"name": class_name},
        },
    }


def _dataframe_to_geojson(df: pd.DataFrame, prob_cols: list[str]) -> dict:
    """Convert a dataframe of tiles to GeoJSON format."""
    features = df.apply(_row_to_geojson, axis=1, prob_cols=prob_cols)
    return {
        "type": "FeatureCollection",
        "features": features.tolist(),
    }


def make_geojson(results_dir: Path, csv: str) -> None:
    
    filename = csv.name.split(".")[0]
    df = pd.read_csv(csv)
    prob_cols = [col for col in df.columns.tolist() if col.startswith("prob_")]
    if not prob_cols:
        raise click.ClickException("Did not find any columns with prob_ prefix.")
    geojson = _dataframe_to_geojson(df, prob_cols)
    with open(
        f"{results_dir}/model-outputs-geojson/{filename}.json", "w"
    ) as f:
        json.dump(geojson, f)


def parallelize_geojson(csvs: list, results_dir: Path, num_workers: int) -> None:
    output = results_dir / "model-outputs-geojson"

    if not results_dir.exists():
        raise FileExistsError(f"results_dir does not exist: {results_dir}")
    if output.exists():
        # raise FileExistsError("Output directory already exists.")
        shutil.rmtree(f"{output}")
    if (
        not (results_dir / "model-outputs-csv").exists()
        and (results_dir / "patches").exists()
    ):
        raise FileExistsError(
            "Model outputs have not been generated yet. Please run model inference."
        )
    if not (results_dir / "model-outputs-csv").exists():
        raise FileExistsError(
            "Expected results_dir to contain a 'model-outputs' directory but it does"
            " not. Please provide the path to the directory that contains"
            " model-outputs, masks, and patches."
        )

    output.mkdir(parents=True, exist_ok=True)

    func = partial(make_geojson, results_dir)
    process_map(func, csvs, max_workers=num_workers)
