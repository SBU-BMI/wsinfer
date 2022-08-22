"""Convert CSV of model outputs to a GeoJSON file.

This GeoJSON file can be loaded into whole slide image viewers like QuPath.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def _box_to_polygon(*, minx: int, miny: int, width: int, height: int):
    """Get coordinates of a box polygon."""
    maxx = minx + width
    maxy = miny + height
    return [(maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny), (maxx, miny)]


def _row_to_geojson(row: pd.Series):
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


def _dataframe_to_geojson(df: pd.DataFrame):
    features = df.apply(_row_to_geojson, axis=1)
    return {
        "type": "FeatureCollection",
        "features": features.tolist(),
    }


def convert(input, output):
    df = pd.read_csv(input)
    geojson = _dataframe_to_geojson(df)
    with open(output, "w") as f:
        json.dump(geojson, f)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Input CSV")
    p.add_argument("output", help="Output GeoJSON file (with .json extension)")
    args = p.parse_args()
    args.input = Path(args.input)
    args.output = Path(args.output).with_suffix(".json")
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.output.exists():
        p.exit(status=0, message=f"Output file exists: {args.output}\n")
    convert(input=args.input, output=args.output)
    print(f"Saved output to {args.output}")
