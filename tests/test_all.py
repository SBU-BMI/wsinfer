import json
from pathlib import Path
import sys

import click
from click.testing import CliRunner
import numpy as np
import pandas as pd
import pytest
import tifffile


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


def test_cli_list():
    from wsinfer.cli.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["list"])
    assert "resnet34" in result.output
    assert "TCGA-BRCA-v1" in result.output
    assert result.exit_code == 0


def test_cli_run_args(tmp_path: Path):
    """Test that (model and weights) or config is required."""
    from wsinfer.cli.cli import cli

    wsi_dir = tmp_path / "slides"
    wsi_dir.mkdir()

    runner = CliRunner()
    args = [
        "run",
        "--wsi-dir",
        wsi_dir,
        "--results-dir",
        tmp_path / "results",
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


def test_cli_run_and_convert(tiff_image: Path, tmp_path: Path):
    """This is a form of a regression test."""
    from wsinfer.cli.cli import cli

    runner = CliRunner()
    results_dir = tmp_path / "inference"
    result = runner.invoke(
        cli,
        [
            "run",
            "--wsi-dir",
            tiff_image.parent,
            "--model",
            "resnet34",
            "--weights",
            "TCGA-BRCA-v1",
            "--results-dir",
            results_dir,
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

    # Test conversion scripts.
    geojson_dir = results_dir / "geojson"
    result = runner.invoke(cli, ["togeojson", str(results_dir), str(geojson_dir)])
    assert result.exit_code == 0
    with open(geojson_dir / "purple.json") as f:
        d = json.load(f)
    assert len(d["features"]) == 144

    for geojson_row in d["features"]:
        assert geojson_row["type"] == "Feature"
        assert geojson_row["id"] == "PathTileObject"
        assert geojson_row["geometry"]["type"] == "Polygon"

    # Check the probability values.
    assert all(
        np.allclose(dd["properties"]["measurements"][0]["value"], 0.9525967836380004)
        for dd in d["features"]
    )
    assert all(
        np.allclose(dd["properties"]["measurements"][1]["value"], 0.0474032945930957)
        for dd in d["features"]
    )

    # Check the names.
    assert all(
        dd["properties"]["measurements"][0]["name"] == "prob_notumor"
        for dd in d["features"]
    )
    assert all(
        dd["properties"]["measurements"][1]["name"] == "prob_tumor"
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
            tiff_image.parent,
            "--config",
            config,
            "--results-dir",
            results_dir,
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
