"""Convert CSV of model outputs to a GeoJSON file.

This GeoJSON file can be loaded into whole slide image viewers like QuPath.
"""

import json
from pathlib import Path
import typing

import click
import pandas as pd


def _box_to_polygon(
    *, minx: int, miny: int, width: int, height: int
) -> typing.List[typing.Tuple[int, int]]:
    """Get coordinates of a box polygon."""
    maxx = minx + width
    maxy = miny + height
    return [(maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny), (maxx, miny)]


def _row_to_geojson(row: pd.Series) -> typing.Dict:
    minx, miny, width, height = row["minx"], row["miny"], row["width"], row["height"]
    coords = _box_to_polygon(minx=minx, miny=miny, width=width, height=height)
    return {
        "type": "Feature",
        "id": "PathTileObject",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
        "properties": {
            "isLocked": True,
            "measurements": [{"name": "Probability", "value": row["cls1"]}],
            "classification": {"name": "Tumor"},
        },
    }


def _dataframe_to_geojson(df: pd.DataFrame) -> typing.Dict:
    features = df.apply(_row_to_geojson, axis=1)
    return {
        "type": "FeatureCollection",
        "features": features.tolist(),
    }


def convert(input, output) -> None:
    df = pd.read_csv(input)
    geojson = _dataframe_to_geojson(df)
    with open(output, "w") as f:
        json.dump(geojson, f)


def _version() -> str:
    """Closure to return version (avoid possibility of a circular import)."""
    from . import __version__

    return __version__


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument(
    "output",
    type=click.Path(dir_okay=False, exists=False, writable=True, path_type=Path),
)
@click.version_option(version=_version())
def cli(*, input: Path, output: Path):
    """Convert CSV of patch predictions to a GeoJSON file.

    GeoJSON files can be used with pathology viewers like QuPath.

    INPUT       Path to input CSV

    OUTPUT      Path to output GeoJSON file (with .json extension)
    """
    click.echo(f"Reading CSV: {input}")
    convert(input=input, output=output)
    click.secho(f"Saved output to {output}", fg="green")
