from __future__ import annotations

import sys
import json
from pathlib import Path


def add_image_and_geojson(
    qupath_proj: object,
    *,
    image_path: Path | str,
    geojson_path: Path | str,
) -> None:
    with open(geojson_path) as f:
        try:
            geojson_features = json.load(f)["features"]
        except Exception as e:
            print(f"Unable to find features key:: {e}")

    entry = qupath_proj.add_image(image_path)
    try:
        entry.hierarchy.load_geojson(geojson_features)
    except Exception as e:
        print(f"Failed to run load_geojson function with error:: {e}")


def make_qupath_project(wsi_dir: Path, results_dir: Path) -> None:
    try:
        from paquo.projects import QuPathProject

        HAS_PAQUO = True
    except Exception:
        HAS_PAQUO = False

    if not HAS_PAQUO:
        print(
            """Cannot find QuPath.
QuPath is required to use this functionality but it cannot be found.
If QuPath is installed, please use define the environment variable
PAQUO_QUPATH_DIR with the location of the QuPath installation.
If QuPath is not installed, please install it from https://qupath.github.io/."""
        )
        sys.exit(1)
    else:
        print("Found QuPath successfully!")
        QUPATH_PROJECT_DIRECTORY = results_dir / "model-outputs-qupath"

        csv_files = list((results_dir / "model-outputs-csv").glob("*.csv"))
        slides_and_geojsons = []

        for csv_file in csv_files:
            file_name = csv_file.stem

            json_file = results_dir / "model-outputs-geojson" / (file_name + ".json")
            image_file = wsi_dir / (file_name + ".svs")

            if json_file.exists() and image_file.exists():
                matching_pair = (image_file, json_file)
                slides_and_geojsons.append(matching_pair)
            else:
                print(f"Skipping CSV: {csv_file.name} (No corresponding JSON)")

        with QuPathProject(QUPATH_PROJECT_DIRECTORY, mode="w") as qp:
            for image_path, geojson_path in slides_and_geojsons:
                try:
                    add_image_and_geojson(
                        qp, image_path=image_path, geojson_path=geojson_path
                    )
                except Exception as e:
                    print(f"Failed to add image/geojson with error:: {e}")
        print("Successfully created QuPath Project!")
