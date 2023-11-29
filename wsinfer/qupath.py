from __future__ import annotations

import sys
import json
from pathlib import Path
import subprocess
import os
import toml
import paquo
from natsort import natsorted #add in dependencies?

def configure_qupath():

    try:
        from paquo.projects import QuPathProject
    except Exception as e:
        print(f"Couldn't find Qupath project with error: {e}")

        choice = input("""QuPath can be configured by setting the 'qupath_dir' field in '.paquo.toml'.
        You can also manually enter the path to your local QuPath installation.
        Do you want to enter manually? (Y[yes] or n[no]): 
        """)


        if choice is None or choice != 'Y':
            pass
        elif choice == 'Y':
            ### Converting the string to Path doesnt work. Gives TypeError: str expected, not PosixPath
            qupath_directory = input("Please enter the exact path where QuPath is installed: ")

            if Path(qupath_directory).exists():
                os.environ["PAQUO_QUPATH_DIR"] = str(qupath_directory) # setting the env var 
                paquo.settings.reload() # Reloading 
            else:
                print(f"QuPath Directory not found. Try again!")
                sys.exit(1)

def add_image_and_geojson(
    qupath_proj: QuPathProject, *, image_path: Path | str, geojson_path: Path | str
) -> None:
    with open(geojson_path) as f:
        # FIXME: check that a 'features' key is present and raise a useful error if not
        geojson_features = json.load(f)["features"]

    entry = qupath_proj.add_image(image_path)
    # FIXME: test that the 'load_geojson' function exists. If not, raise a useful error
    entry.hierarchy.load_geojson(geojson_features)  # type: ignore


# Store a list of matched slides and geojson files. Linking the slides and geojson in
# this way prevents a potential mismatch by simply listing directories and relying on
# the order to be the same.


def make_qupath_project(wsi_dir, results_dir):

    configure_qupath() # Sets the environment variable "PAQUO_QUPATH_DIR"
    try:
        from paquo.projects import QuPathProject
    except: 
        print("Unable to find Qupath! Run the program again")
        sys.exit(1)

    print("Found QuPath successfully!")
    QUPATH_PROJECT_DIRECTORY = "QuPathProject"

    csv_list = natsorted([str(file) for file in wsi_dir.iterdir() if file.is_file()])
    json_list = natsorted([str(file) for file in Path(f"{results_dir}/model-outputs-geojson").iterdir() if file.is_file()])

    slides_and_geojsons = [
        (csv, json) for csv, json in zip(csv_list, json_list)
    ]
    with QuPathProject(QUPATH_PROJECT_DIRECTORY, mode="w") as qp:
        for image_path, geojson_path in slides_and_geojsons:
            try:
                add_image_and_geojson(qp, image_path=image_path, geojson_path=geojson_path)
            except Exception as e:
                print(f"Failed to add image/geojson with error:: {e}")
    print("Successfully created QuPath Project!")