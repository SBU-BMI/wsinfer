from pathlib import Path

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

    try:
        resolutionunit = tifffile.RESUNIT.CENTIMETER
    except AttributeError:
        resolutionunit = 3  # for python 3.7 compat

    tifffile.imwrite(
        path,
        data=x,
        compression="zlib",
        tile=(256, 256),
        # 0.25 micrometers per pixel.
        resolution=(40000, 40000),
        resolutionunit=resolutionunit,
    )
    return path


def test_cli_list():
    from wsinfer.cli.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["list"])
    assert "resnet34" in result.output
    assert "TCGA-BRCA-v1" in result.output
    assert result.exit_code == 0


def test_cli_run(tiff_image: Path, tmp_path: Path):
    from wsinfer.cli.cli import cli

    runner = CliRunner()
    results_dir = tmp_path / "inference"
    result = runner.invoke(
        cli,
        [
            "run",
            "--wsi_dir",
            tiff_image.parent,
            "--model",
            "resnet34",
            "--weights",
            "TCGA-BRCA-v1",
            "--results_dir",
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
