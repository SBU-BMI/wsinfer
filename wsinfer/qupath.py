from __future__ import annotations

import json
from pathlib import Path
import subprocess
import toml
import os

# try:
#     subprocess.run(f"python -m paquo config -l -o {Path.cwd()}", shell=True, check=True)
#     print("Command executed successfully.")
#     data = toml.load(".paquo.toml")
#     data["qupath_dir"] = "/home/sggat/QuPath"

#     f = open(".paquo.toml",'w')
#     toml.dump(data, f)
#     f.close()
# except subprocess.CalledProcessError as e:
#     print(f"Error running the command: {e}")

def configure_qupath():

    choice = input("""QuPath can be configured by setting the 'qupath_dir' 
                    field in '.paquo.toml'. You can also manually enter the path
                    to your local QuPath installation. Do you want to enter manually?
                    Y[yes] or n[no]:
       """)

    if choice is None or choice == 'n':
        pass
    elif choice == 'Y':
        ### Converting the string to Path doesnt work. Gives TypeError: str expected, not PosixPath
        qupath_directory = input("Please enter the exact path where QuPath is installed: ")

        try: 
            if Path(qupath_directory).exists():
                os.environ["PAQUO_QUPATH_DIR"] = qupath_directory
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            print(f"QuPath Directory not found. Try again!")

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

    configure_qupath()
    from paquo.projects import QuPathProject
    QUPATH_PROJECT_DIRECTORY = "QuPathProject"

    slides_and_geojsons = [
    ("/home/sggat/wsinfer/wsinfer/SlideImages/CMU-1.svs", "/home/sggat/wsinfer/wsinfer/Results/model-outputs-geojson/CMU-1.json"),
    ("/home/sggat/wsinfer/wsinfer/SlideImages/CMU-2.svs", "/home/sggat/wsinfer/wsinfer/Results/model-outputs-geojson/CMU-2.json"),
    ("/home/sggat/wsinfer/wsinfer/SlideImages/CMU-3.svs", "/home/sggat/wsinfer/wsinfer/Results/model-outputs-geojson/CMU-3.json"),
]
    with QuPathProject(QUPATH_PROJECT_DIRECTORY, mode="w") as qp:
        for image_path, geojson_path in slides_and_geojsons:
            try:
                add_image_and_geojson(qp, image_path=image_path, geojson_path=geojson_path)
            except Exception as e:
                print(f"Failed to add image/geojson with error:: {e}")