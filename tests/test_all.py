import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import List

from click.testing import CliRunner
import geojson as geojsonlib
import h5py
import numpy as np
import pandas as pd
import pytest
import tifffile
import torch
import yaml

from wsinfer import get_model_weights
from wsinfer import list_all_models_and_weights


@pytest.fixture
def tiff_image(tmp_path: Path) -> Path:
    x = np.empty((4096, 4096, 3), dtype="uint8")
    x[...] = [160, 32, 240]  # rgb for purple
    path = Path(tmp_path / "images" / "purple.tif")
    path.parent.mkdir(exist_ok=True)

    if sys.version_info >= (3, 8):
        tifffile.imwrite(
            path,
            data=x,
            compression="zlib",
            tile=(256, 256),
            # 0.25 micrometers per pixel.
            resolution=(40000, 40000),
            resolutionunit=tifffile.RESUNIT.CENTIMETER,
        )
    else:
        # Earlier versions of tifffile do not have resolutionunit kwarg.
        tifffile.imwrite(
            path,
            data=x,
            compression="zlib",
            tile=(256, 256),
            # 0.25 micrometers per pixel.
            resolution=(40000, 40000, "CENTIMETER"),
        )

    return path


def test_cli_list(tmp_path: Path):
    from wsinfer.cli.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["list"])
    assert "resnet34" in result.output
    assert "TCGA-BRCA-v1" in result.output
    assert result.exit_code == 0

    # Test of WSINFER_PATH registration... check that the models appear in list.
    # Test of single WSINFER_PATH.
    config_root_single = tmp_path / "configs-single"
    config_root_single.mkdir()
    configs = [
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        dict(
            version="1.0",
            name="foo2",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
    ]
    for i, config in enumerate(configs):
        with open(config_root_single / f"{i}.yaml", "w") as f:
            yaml.safe_dump(config, f)

    ret = subprocess.run(
        [sys.executable, "-m", "wsinfer", "list"],
        capture_output=True,
        env=dict(WSINFER_PATH=str(config_root_single)),
    )
    assert ret.returncode == 0
    output = ret.stdout.decode()
    assert configs[0]["name"] in output  # type: ignore
    assert configs[0]["architecture"] in output  # type: ignore
    assert configs[1]["name"] in output  # type: ignore
    assert configs[1]["architecture"] in output  # type: ignore
    # Negative control.
    ret = subprocess.run([sys.executable, "-m", "wsinfer", "list"], capture_output=True)
    assert configs[0]["name"] not in ret.stdout.decode()  # type: ignore
    del config_root_single, output, ret, config

    # Test of WSINFER_PATH registration... check that the models appear in list.
    # Test of multiple WSINFER_PATH.
    config_root = tmp_path / "configs"
    config_root.mkdir()
    config_paths = [config_root / "0", config_root / "1"]
    for i, config in enumerate(configs):
        config_paths[i].mkdir()
        with open(config_paths[i] / f"{i}.yaml", "w") as f:
            yaml.safe_dump(config, f)

    ret = subprocess.run(
        [sys.executable, "-m", "wsinfer", "list"],
        capture_output=True,
        env=dict(WSINFER_PATH=":".join(str(c) for c in config_paths)),
    )
    assert ret.returncode == 0
    output = ret.stdout.decode()
    assert configs[0]["name"] in output  # type: ignore
    assert configs[0]["architecture"] in output  # type: ignore
    assert configs[1]["name"] in output  # type: ignore
    assert configs[1]["architecture"] in output  # type: ignore
    ret = subprocess.run([sys.executable, "-m", "wsinfer", "list"], capture_output=True)
    assert configs[0]["name"] not in ret.stdout.decode()  # type: ignore


def test_cli_run_args(tmp_path: Path):
    """Test that (model and weights) or config is required."""
    from wsinfer.cli.cli import cli

    wsi_dir = tmp_path / "slides"
    wsi_dir.mkdir()

    runner = CliRunner()
    args = [
        "run",
        "--wsi-dir",
        str(wsi_dir),
        "--results-dir",
        str(tmp_path / "results"),
    ]
    # No model, weights, or config.
    result = runner.invoke(cli, args)
    assert result.exit_code != 0
    assert "one of (model and weights) or config is required." in result.output

    # Only one of model and weights.
    result = runner.invoke(cli, [*args, "--model", "resnet34"])
    assert result.exit_code != 0
    assert "model and weights must both be set if one is set." in result.output
    result = runner.invoke(cli, [*args, "--weights", "TCGA-BRCA-v1"])
    assert result.exit_code != 0
    assert "model and weights must both be set if one is set." in result.output

    # config and model
    result = runner.invoke(cli, [*args, "--config", __file__, "--model", "resnet34"])
    assert result.exit_code != 0
    assert "model and weights are mutually exclusive with config." in result.output
    # config and weights
    result = runner.invoke(
        cli, [*args, "--config", __file__, "--weights", "TCGA-BRCA-v1"]
    )
    assert result.exit_code != 0
    assert "model and weights are mutually exclusive with config." in result.output


@pytest.mark.parametrize(
    [
        "model",
        "weights",
        "class_names",
        "expected_probs",
        "expected_patch_size",
        "expected_num_patches",
    ],
    [
        # Resnet34 TCGA-BRCA-v1
        (
            "resnet34",
            "TCGA-BRCA-v1",
            ["notumor", "tumor"],
            [0.9525967836380005, 0.04740329459309578],
            350,
            144,
        ),
        # Resnet34 TCGA-LUAD-v1
        (
            "resnet34",
            "TCGA-LUAD-v1",
            ["lepidic", "benign", "acinar", "micropapillary", "mucinous", "solid"],
            [
                0.012793001718819141,
                0.9792948961257935,
                0.0050891609862446785,
                0.0003837027761619538,
                0.0006556913140229881,
                0.0017834495520219207,
            ],
            700,
            36,
        ),
        # Resnet34 TCGA-PRAD-v1
        (
            "resnet34",
            "TCGA-PRAD-v1",
            ["grade3", "grade4+5", "benign"],
            [0.0010944147361442447, 3.371985076228157e-05, 0.9988718628883362],
            350,
            144,
        ),
        # Inception_v4 TCGA-BRCA-v1
        (
            "inception_v4",
            "TCGA-BRCA-v1",
            ["notumor", "tumor"],
            [0.9564113020896912, 0.043588679283857346],
            350,
            144,
        ),
        # Inceptionv4nobn TCGA-TILs-v1
        (
            "inception_v4nobn",
            "TCGA-TILs-v1",
            ["notils", "tils"],
            [1.0, 3.427359524660334e-12],
            200,
            441,
        ),
        # Vgg16mod TCGA-BRCA-v1
        (
            "vgg16mod",
            "TCGA-BRCA-v1",
            ["notumor", "tumor"],
            [0.9108286499977112, 0.089171402156353],
            350,
            144,
        ),
        # Preactresnet34 TCGA-PAAD-v1
        (
            "preactresnet34",
            "TCGA-PAAD-v1",
            ["tumor"],
            [0.01446483],
            2100,
            4,
        ),
    ],
)
@pytest.mark.parametrize("speedup", [False, True])
def test_cli_run_regression(
    model: str,
    weights: str,
    class_names: List[str],
    expected_probs: List[float],
    expected_patch_size: int,
    expected_num_patches: int,
    speedup: bool,
    tiff_image: Path,
    tmp_path: Path,
):
    """A regression test of the command 'wsinfer run', using all registered models."""
    from wsinfer.cli.cli import cli

    runner = CliRunner()
    results_dir = tmp_path / "inference"
    result = runner.invoke(
        cli,
        [
            "run",
            "--wsi-dir",
            str(tiff_image.parent),
            "--model",
            model,
            "--weights",
            weights,
            "--results-dir",
            str(results_dir),
            "--speedup" if speedup else "--no-speedup",
        ],
    )
    assert result.exit_code == 0
    assert (results_dir / "model-outputs").exists()
    df = pd.read_csv(results_dir / "model-outputs" / "purple.csv")
    class_prob_cols = [f"prob_{c}" for c in class_names]
    assert df.columns.tolist() == [
        "slide",
        "minx",
        "miny",
        "width",
        "height",
        *class_prob_cols,
    ]
    # TODO: test the metadata.json file as well.
    assert df.shape[0] == expected_num_patches
    assert (df.loc[:, "slide"] == str(tiff_image)).all()
    assert (df.loc[:, "width"] == expected_patch_size).all()
    assert (df.loc[:, "height"] == expected_patch_size).all()
    # Test probs.
    for col, col_prob in zip(class_names, expected_probs):
        col = f"prob_{col}"
        assert np.allclose(df.loc[:, col], col_prob)

    # Test that metadata path exists.
    metadata_paths = list(results_dir.glob("run_metadata_*.json"))
    assert len(metadata_paths) == 1
    metadata_path = metadata_paths[0]
    assert metadata_path.exists()
    with open(metadata_path) as f:
        meta = json.load(f)
    assert meta.keys() == {"model_weights", "runtime", "timestamp"}
    assert meta["model_weights"]["name"] == weights
    assert meta["model_weights"]["architecture"] == model
    assert meta["model_weights"]["class_names"] == class_names
    assert meta["runtime"]["python_executable"] == sys.executable
    assert meta["runtime"]["python_version"] == platform.python_version()
    assert meta["timestamp"]
    del metadata_path, meta

    # Test conversion scripts.
    geojson_dir = results_dir / "geojson"
    result = runner.invoke(cli, ["togeojson", str(results_dir), str(geojson_dir)])
    assert result.exit_code == 0
    with open(geojson_dir / "purple.json") as f:
        d: geojsonlib.GeoJSON = geojsonlib.load(f)
    assert d.is_valid, "geojson not valid!"
    assert len(d["features"]) == expected_num_patches

    for geojson_row in d["features"]:
        assert geojson_row["type"] == "Feature"
        assert geojson_row["id"] == "PathTileObject"
        assert geojson_row["geometry"]["type"] == "Polygon"

    # Check the probability values.
    for i, prob in enumerate(expected_probs):
        # names have the prefix "prob_".
        assert all(
            dd["properties"]["measurements"][i]["name"] == class_prob_cols[i]
            for dd in d["features"]
        )
        assert all(
            np.allclose(dd["properties"]["measurements"][i]["value"], prob)
            for dd in d["features"]
        )

    # Check the coordinate values.
    for df_row, geojson_row in zip(df.itertuples(), d["features"]):
        maxx = df_row.minx + df_row.width
        maxy = df_row.miny + df_row.height
        df_coords = [
            [maxx, df_row.miny],
            [maxx, maxy],
            [df_row.minx, maxy],
            [df_row.minx, df_row.miny],
            [maxx, df_row.miny],
        ]
        assert [df_coords] == geojson_row["geometry"]["coordinates"]


@pytest.mark.xfail
def test_convert_to_sbu():
    # TODO: create a synthetic output and then convert it. Check that it is valid.
    assert False


def test_cli_run_from_config(tiff_image: Path, tmp_path: Path):
    """This is a form of a regression test."""
    import wsinfer
    from wsinfer.cli.cli import cli

    # Use config for resnet34 TCGA-BRCA-v1 weights.
    config = Path(wsinfer.__file__).parent / "modeldefs" / "resnet34_tcga-brca-v1.yaml"
    assert config.exists()

    runner = CliRunner()
    results_dir = tmp_path / "inference"
    result = runner.invoke(
        cli,
        [
            "run",
            "--wsi-dir",
            str(tiff_image.parent),
            "--config",
            str(config),
            "--results-dir",
            str(results_dir),
        ],
    )
    assert result.exit_code == 0
    assert (results_dir / "model-outputs").exists()
    df = pd.read_csv(results_dir / "model-outputs" / "purple.csv")
    assert df.columns.tolist() == [
        "slide",
        "minx",
        "miny",
        "width",
        "height",
        "prob_notumor",
        "prob_tumor",
    ]
    assert (df.loc[:, "slide"] == str(tiff_image)).all()
    assert (df.loc[:, "width"] == 350).all()
    assert (df.loc[:, "height"] == 350).all()
    assert (df.loc[:, "width"] == 350).all()
    assert np.allclose(df.loc[:, "prob_notumor"], 0.9525967836380005)
    assert np.allclose(df.loc[:, "prob_tumor"], 0.04740329459309578)


@pytest.mark.parametrize(
    "modeldef",
    [
        [],
        {},
        dict(name="foo", architecture="resnet34"),
        # Missing url
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            # url="foo",
            # url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # missing url_file_name when url is given
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            # url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # url and file used together
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            file=__file__,
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # nonexistent file
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            file="path/to/fake/file",
            # url="foo",
            # url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # num_classes missing
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            # num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # num classes not equal to len of class names
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=2,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # transform missing
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            # transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # transform.resize_size missing
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # transform.mean missing
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # transform.std missing
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # transform.resize_size non int
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=0.5, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # transform.resize_size non int
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(
                resize_size=[100, 100], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # transform.mean not a list of three floats
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # transform.mean not a list of three floats
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[1, 1, 1], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # transform.mean not a list of three floats
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=0.5, std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # transform.std not a list of three floats
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, std=[0.5], mean=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # transform.std not a list of three floats
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, std=[1, 1, 1], mean=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # transform.std not a list of three floats
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, std=0.5, mean=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # invalid patch_size_pixels -- list
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=[350],
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # invalid patch_size_pixels -- float
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350.0,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # invalid patch_size_pixels -- negative
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=-100,
            spacing_um_px=0.25,
            class_names=["tumor"],
        ),
        # invalid spacing_um_px -- zero
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0,
            class_names=["tumor"],
        ),
        # invalid spacing_um_px -- list
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=[0.25],
            class_names=["tumor"],
        ),
        # invalid class_names -- str
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names="t",
        ),
        # invalid class_names -- len not equal to num_classes
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["tumor", "nontumor"],
        ),
        # invalid class_names -- not list of str
        dict(
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=[1],
        ),
        # unknown key
        dict(
            fakekey="foobar",
            version="1.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["foo"],
        ),
        # version != '1.0'
        dict(
            version="2.0",
            name="foo",
            architecture="resnet34",
            url="foo",
            url_file_name="foo",
            num_classes=1,
            transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            patch_size_pixels=350,
            spacing_um_px=0.25,
            class_names=["foo"],
        ),
    ],
)
def test_invalid_modeldefs(modeldef, tmp_path: Path):
    from wsinfer._modellib.models import Weights

    path = tmp_path / "foobar.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(modeldef, f)

    with pytest.raises(Exception):
        Weights.from_yaml(path)


def test_valid_modeldefs(tmp_path: Path):
    from wsinfer._modellib.models import Weights

    weights_file = tmp_path / "weights.pt"
    modeldef = dict(
        version="1.0",
        name="foo",
        architecture="resnet34",
        file=str(weights_file),
        num_classes=2,
        transform=dict(resize_size=224, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        patch_size_pixels=350,
        spacing_um_px=0.25,
        class_names=["foo", "bar"],
    )
    path = tmp_path / "foobar.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(modeldef, f)

    with pytest.raises(FileNotFoundError):
        Weights.from_yaml(path)

    weights_file.touch()
    assert Weights.from_yaml(path)


def test_model_registration(tmp_path: Path):
    from wsinfer._modellib import models

    # Test that registering duplicate weights will error.
    d = dict(
        version="1.0",
        name="foo",
        architecture="resnet34",
        url="foo",
        url_file_name="foo",
        num_classes=1,
        transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        patch_size_pixels=350,
        spacing_um_px=0.25,
        class_names=["foo"],
    )
    path = tmp_path / "foobar.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(d, f)
    path = tmp_path / "foobardup.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(d, f)

    with pytest.raises(models.DuplicateModelWeights):
        models.register_model_weights(tmp_path)

    # Test that registering models will put them in the _known_model_weights object.
    path = tmp_path / "configs" / "foobar.yaml"
    path.parent.mkdir()
    d = dict(
        version="1.0",
        name="foo2",
        architecture="resnet34",
        url="foo",
        url_file_name="foo",
        num_classes=1,
        transform=dict(resize_size=299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        patch_size_pixels=350,
        spacing_um_px=0.25,
        class_names=["foo"],
    )
    with open(path, "w") as f:
        yaml.safe_dump(d, f)
    models.register_model_weights(path.parent)
    assert (d["architecture"], d["name"]) in models._known_model_weights.keys()
    assert all(
        isinstance(m, models.Weights) for m in models._known_model_weights.values()
    )


@pytest.mark.parametrize(
    ["patch_size", "patch_spacing"],
    [(256, 0.25), (256, 0.50), (350, 0.25)],
)
def test_patch_cli(
    patch_size: int, patch_spacing: float, tmp_path: Path, tiff_image: Path
):
    from wsinfer.cli.cli import cli

    orig_slide_width = 4096
    orig_slide_height = 4096
    orig_slide_spacing = 0.25

    runner = CliRunner()
    savedir = tmp_path / "savedir"
    result = runner.invoke(
        cli,
        [
            "patch",
            "--source",
            str(tiff_image.parent),
            "--save-dir",
            str(savedir),
            "--patch-size",
            str(patch_size),
            "--patch-spacing",
            str(patch_spacing),
        ],
    )
    assert result.exit_code == 0
    stem = tiff_image.stem
    assert (savedir / "masks" / f"{stem}.jpg").exists()
    assert (savedir / "patches" / f"{stem}.h5").exists()
    assert (savedir / "process_list_autogen.csv").exists()
    assert (savedir / "stitches" / f"{stem}.jpg").exists()

    expected_patch_size = round(patch_size * patch_spacing / orig_slide_spacing)
    expected_num_patches = math.ceil(4096 / expected_patch_size) ** 2
    expected_coords = []
    for x in range(0, orig_slide_width, expected_patch_size):
        for y in range(0, orig_slide_height, expected_patch_size):
            expected_coords.append([x, y])
    expected_coords_arr = np.array(expected_coords)

    with h5py.File(savedir / "patches" / f"{stem}.h5") as f:
        assert f["/coords"].attrs["patch_size"] == expected_patch_size
        coords = f["/coords"][()]
    assert coords.shape == (expected_num_patches, 2)
    assert np.array_equal(expected_coords_arr, coords)


@pytest.mark.parametrize(["model_name", "weights_name"], list_all_models_and_weights())
def test_jit_compile(model_name: str, weights_name: str):
    import time
    from wsinfer._modellib.run_inference import jit_compile

    w = get_model_weights(model_name, weights_name)
    size = w.transform.resize_size
    x = torch.ones(20, 3, size, size, dtype=torch.float32)
    model = w.load_model()
    model.eval()
    NUM_SAMPLES = 1
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(NUM_SAMPLES):
            out_nojit = model(x).detach().cpu()
        time_nojit = time.perf_counter() - t0
    model_nojit = model
    model = jit_compile(model)
    if model is model_nojit:
        pytest.skip("Failed to compile model (would use original model)")
    with torch.no_grad():
        model(x).detach().cpu()  # run it once to compile
        t0 = time.perf_counter()
        for _ in range(NUM_SAMPLES):
            out_jit = model(x).detach().cpu()
        time_yesjit = time.perf_counter() - t0

    assert torch.allclose(out_nojit, out_jit)
    if time_nojit < time_yesjit:
        pytest.skip(
            "JIT-compiled model was SLOWER than original: "
            f"jit={time_yesjit:0.3f} vs nojit={time_nojit:0.3f}"
        )


def test_issue_89():
    """Do not fail if 'git' is not installed."""
    from wsinfer.cli.infer import _get_info_for_save

    w = get_model_weights("resnet34", "TCGA-BRCA-v1")
    d = _get_info_for_save(w)
    assert d
    assert "git" in d["runtime"]
    assert d["runtime"]["git"]
    assert d["runtime"]["git"]["git_remote"]
    assert d["runtime"]["git"]["git_branch"]

    # Test that _get_info_for_save does not fail if git is not found.
    orig_path = os.environ["PATH"]
    try:
        os.environ["PATH"] = ""
        w = get_model_weights("resnet34", "TCGA-BRCA-v1")
        d = _get_info_for_save(w)
        assert d
        assert "git" in d["runtime"]
        assert d["runtime"]["git"] is None
    finally:
        os.environ["PATH"] = orig_path  # reset path


def test_issue_94(tmp_path: Path, tiff_image: Path):
    """Gracefully handle unreadable slides."""
    from wsinfer.cli.cli import cli

    # We have a valid tiff in 'tiff_image.parent'. We put in an unreadable file too.
    badpath = tiff_image.parent / "bad.svs"
    badpath.touch()

    runner = CliRunner()
    results_dir = tmp_path / "inference"
    result = runner.invoke(
        cli,
        [
            "run",
            "--wsi-dir",
            str(tiff_image.parent),
            "--model",
            "resnet34",
            "--weights",
            "TCGA-BRCA-v1",
            "--results-dir",
            str(results_dir),
        ],
    )
    # Important part is that we run through all of the files, despite the unreadble
    # file.
    assert result.exit_code == 0
    assert results_dir.joinpath("model-outputs").joinpath("purple.csv").exists()
    assert not results_dir.joinpath("model-outputs").joinpath("bad.csv").exists()


def test_issue_97(tmp_path: Path, tiff_image: Path):
    """Write a run_metadata file per run."""
    from wsinfer.cli.cli import cli

    runner = CliRunner()
    results_dir = tmp_path / "inference"
    result = runner.invoke(
        cli,
        [
            "run",
            "--wsi-dir",
            str(tiff_image.parent),
            "--model",
            "resnet34",
            "--weights",
            "TCGA-BRCA-v1",
            "--results-dir",
            str(results_dir),
        ],
    )
    assert result.exit_code == 0
    metas = list(results_dir.glob("run_metadata_*.json"))
    assert len(metas) == 1

    # Run again...
    result = runner.invoke(
        cli,
        [
            "run",
            "--wsi-dir",
            str(tiff_image.parent),
            "--model",
            "resnet34",
            "--weights",
            "TCGA-BRCA-v1",
            "--results-dir",
            str(results_dir),
        ],
    )
    assert result.exit_code == 0
    metas = list(results_dir.glob("run_metadata_*.json"))
    assert len(metas) == 2
