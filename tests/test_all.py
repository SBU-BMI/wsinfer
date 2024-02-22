from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from unittest.mock import patch as mock_patch, MagicMock

import geojson as geojsonlib
import h5py
import numpy as np
import pandas as pd
import pytest
import tifffile
import torch
from click.testing import CliRunner

from wsinfer.cli.cli import cli
from wsinfer.cli.infer import _get_info_for_save
from wsinfer.modellib.models import get_pretrained_torch_module
from wsinfer.modellib.models import get_registered_model
from wsinfer.modellib.run_inference import jit_compile
from wsinfer.wsi import HAS_OPENSLIDE
from wsinfer.wsi import HAS_TIFFSLIDE


@pytest.fixture
def tiff_image(tmp_path: Path) -> Path:
    x = np.empty((4096, 4096, 3), dtype="uint8")
    x[...] = [160, 32, 240]  # rgb for purple
    path = Path(tmp_path / "images" / "purple.tif")
    path.parent.mkdir(exist_ok=True)

    tifffile.imwrite(
        path,
        data=x,
        compression="zlib",
        tile=(256, 256),
        # 0.25 micrometers per pixel.
        resolution=(40_000, 40_000),
        resolutionunit=tifffile.RESUNIT.CENTIMETER,
    )

    return path


# The reference data for this test was made using a patched version of wsinfer 0.3.6.
# The patches fixed an issue when calculating strides and added padding to images.
# Large-image (which was the backend in 0.3.6) did not pad images and would return
# tiles that were not fully the requested width and height.
@pytest.mark.parametrize(
    "model",
    [
        "breast-tumor-resnet34.tcga-brca",
        "lung-tumor-resnet34.tcga-luad",
        "pancancer-lymphocytes-inceptionv4.tcga",
        "pancreas-tumor-preactresnet34.tcga-paad",
        "prostate-tumor-resnet34.tcga-prad",
    ],
)
@pytest.mark.parametrize("speedup", [False, True])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "openslide",
            marks=pytest.mark.skipif(
                not HAS_OPENSLIDE, reason="OpenSlide not available"
            ),
        ),
        pytest.param(
            "tiffslide",
            marks=pytest.mark.skipif(
                not HAS_TIFFSLIDE, reason="TiffSlide not available"
            ),
        ),
    ],
)
def test_cli_run_with_registered_models(
    model: str,
    speedup: bool,
    backend: str,
    tiff_image: Path,
    tmp_path: Path,
) -> None:
    """A regression test of the command 'wsinfer run'."""

    reference_csv = Path(__file__).parent / "reference" / model / "purple.csv"
    if not reference_csv.exists():
        raise FileNotFoundError(f"reference CSV not found: {reference_csv}")

    runner = CliRunner()
    results_dir = tmp_path / "inference"
    result = runner.invoke(
        cli,
        [
            "--backend",
            backend,
            "run",
            "--wsi-dir",
            str(tiff_image.parent),
            "--results-dir",
            str(results_dir),
            "--model",
            model,
            "--speedup" if speedup else "--no-speedup",
        ],
    )
    assert result.exit_code == 0
    assert (results_dir / "model-outputs-csv").exists()
    df = pd.read_csv(results_dir / "model-outputs-csv" / "purple.csv")
    df_ref = pd.read_csv(reference_csv)

    assert set(df.columns) == set(df_ref.columns)
    assert df.shape == df_ref.shape
    assert np.array_equal(df["minx"], df_ref["minx"])
    assert np.array_equal(df["miny"], df_ref["miny"])
    assert np.array_equal(df["width"], df_ref["width"])
    assert np.array_equal(df["height"], df_ref["height"])

    prob_cols = df_ref.filter(like="prob_").columns.tolist()
    for prob_col in prob_cols:
        assert np.allclose(
            df[prob_col], df_ref[prob_col], atol=1e-07
        ), f"Column {prob_col} not allclose at atol=1e-07"

    # Test that metadata path exists.
    metadata_paths = list(results_dir.glob("run_metadata_*.json"))
    assert len(metadata_paths) == 1
    metadata_path = metadata_paths[0]
    assert metadata_path.exists()
    with open(metadata_path) as f:
        meta = json.load(f)
    assert set(meta.keys()) == {"model", "runtime", "timestamp"}
    assert "config" in meta["model"]
    assert "huggingface_location" in meta["model"]
    assert model in meta["model"]["huggingface_location"]["repo_id"]
    assert meta["runtime"]["python_executable"] == sys.executable
    assert meta["runtime"]["python_version"] == platform.python_version()
    assert meta["timestamp"]
    del metadata_path, meta

    # Test conversion to geojson.
    geojson_dir = results_dir / "model-outputs-geojson"
    # result = runner.invoke(cli, ["togeojson", str(results_dir), str(geojson_dir)])
    assert result.exit_code == 0
    with open(geojson_dir / "purple.json") as f:
        d: geojsonlib.GeoJSON = geojsonlib.load(f)
    assert d.is_valid, "geojson not valid!"
    assert len(d["features"]) == len(df_ref)

    for geojson_row in d["features"]:
        assert geojson_row["type"] == "Feature"
        isinstance(geojson_row["id"], str)
        assert geojson_row["geometry"]["type"] == "Polygon"
    res = []
    for i, prob_col in enumerate(prob_cols):
        res.append(
            np.array(
                [dd["properties"]["measurements"][prob_col] for dd in d["features"]]
            )
        )
    geojson_probs = np.stack(res, axis=0)
    del res
    assert np.allclose(df[prob_cols].T, geojson_probs)

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


def test_cli_run_with_local_model(tmp_path: Path, tiff_image: Path) -> None:
    model = "breast-tumor-resnet34.tcga-brca"
    reference_csv = Path(__file__).parent / "reference" / model / "purple.csv"
    if not reference_csv.exists():
        raise FileNotFoundError(f"reference CSV not found: {reference_csv}")
    w = get_registered_model(model)

    config = {
        "spec_version": "1.0",
        "architecture": "resnet34",
        "num_classes": 2,
        "class_names": ["Other", "Tumor"],
        "patch_size_pixels": 350,
        "spacing_um_px": 0.25,
        "transform": [
            {"name": "Resize", "arguments": {"size": 224}},
            {"name": "ToTensor"},
            {
                "name": "Normalize",
                "arguments": {
                    "mean": [0.7238, 0.5716, 0.6779],
                    "std": [0.112, 0.1459, 0.1089],
                },
            },
        ],
    }

    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    runner = CliRunner()
    results_dir = tmp_path / "inference"
    result = runner.invoke(
        cli,
        [
            "--backend",
            "tiffslide",
            "run",
            "--wsi-dir",
            str(tiff_image.parent),
            "--results-dir",
            str(results_dir),
            "--model-path",
            w.model_path,
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0
    assert (results_dir / "model-outputs-csv").exists()
    df = pd.read_csv(results_dir / "model-outputs-csv" / "purple.csv")
    df_ref = pd.read_csv(reference_csv)

    assert set(df.columns) == set(df_ref.columns)
    assert df.shape == df_ref.shape
    assert np.array_equal(df["minx"], df_ref["minx"])
    assert np.array_equal(df["miny"], df_ref["miny"])
    assert np.array_equal(df["width"], df_ref["width"])
    assert np.array_equal(df["height"], df_ref["height"])

    prob_cols = df_ref.filter(like="prob_").columns.tolist()
    for prob_col in prob_cols:
        assert np.allclose(
            df[prob_col], df_ref[prob_col], atol=1e-07
        ), f"Column {prob_col} not allclose at atol=1e-07"


def test_cli_run_no_model_or_config(tmp_path: Path) -> None:
    """Test that --model or (--config and --model-path) is required."""
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
    assert "one of --model or (--config and --model-path) is required" in result.output


def test_cli_run_model_and_config(tmp_path: Path) -> None:
    """Test that (model and weights) or config is required."""
    wsi_dir = tmp_path / "slides"
    wsi_dir.mkdir()

    fake_config = tmp_path / "foobar.json"
    fake_config.touch()
    fake_model_path = tmp_path / "foobar.pt"
    fake_model_path.touch()

    runner = CliRunner()
    args = [
        "run",
        "--wsi-dir",
        str(wsi_dir),
        "--results-dir",
        str(tmp_path / "results"),
        "--model",
        "colorectal-tiatoolbox-resnet50.kather100k",
        "--model-path",
        str(fake_model_path),
        "--config",
        str(fake_config),
    ]
    # No model, weights, or config.
    result = runner.invoke(cli, args)
    assert result.exit_code != 0
    assert (
        "--config and --model-path are mutually exclusive with --model" in result.output
    )


@pytest.mark.xfail
def test_convert_to_sbu() -> None:
    # TODO: create a synthetic output and then convert it. Check that it is valid.
    assert False


@pytest.mark.parametrize(
    ["patch_size", "patch_spacing"],
    [(256, 0.25), (256, 0.50), (350, 0.25), (100, 0.3), (100, 0.5)],
)
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "openslide",
            marks=pytest.mark.skipif(
                not HAS_OPENSLIDE, reason="OpenSlide not available"
            ),
        ),
        pytest.param(
            "tiffslide",
            marks=pytest.mark.skipif(
                not HAS_TIFFSLIDE, reason="TiffSlide not available"
            ),
        ),
    ],
)
def test_patch_cli(
    patch_size: int,
    patch_spacing: float,
    backend: str,
    tmp_path: Path,
    tiff_image: Path,
) -> None:
    """Test of 'wsinfer patch'."""
    orig_slide_size = 4096
    orig_slide_spacing = 0.25

    runner = CliRunner()
    savedir = tmp_path / "savedir"
    result = runner.invoke(
        cli,
        [
            "--backend",
            backend,
            "patch",
            "--wsi-dir",
            str(tiff_image.parent),
            "--results-dir",
            str(savedir),
            "--patch-size-px",
            str(patch_size),
            "--patch-spacing-um-px",
            str(patch_spacing),
        ],
    )
    assert result.exit_code == 0
    stem = tiff_image.stem
    assert (savedir / "masks" / f"{stem}.jpg").exists()
    assert (savedir / "patches" / f"{stem}.h5").exists()

    expected_patch_size = round(patch_size * patch_spacing / orig_slide_spacing)
    sqrt_expected_num_patches = round(orig_slide_size / expected_patch_size)
    expected_num_patches = sqrt_expected_num_patches**2

    expected_coords = []
    for x in range(0, orig_slide_size, expected_patch_size):
        for y in range(0, orig_slide_size, expected_patch_size):
            # Patch is kept if centroid is inside.
            if (
                x + expected_patch_size // 2 <= orig_slide_size
                and y + expected_patch_size // 2 <= orig_slide_size
            ):
                expected_coords.append([x, y])
    assert len(expected_coords) == expected_num_patches
    with h5py.File(savedir / "patches" / f"{stem}.h5") as f:
        assert f["/coords"].attrs["patch_size"] == expected_patch_size
        coords = f["/coords"][()]
    assert coords.shape == (expected_num_patches, 2)
    assert np.array_equal(expected_coords, coords)


# FIXME: parametrize this test across our models.
def test_jit_compile() -> None:
    w = get_registered_model("breast-tumor-resnet34.tcga-brca")
    model = get_pretrained_torch_module(w)

    x = torch.ones(20, 3, 224, 224, dtype=torch.float32)
    model.eval()
    NUM_SAMPLES = 1
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(NUM_SAMPLES):
            out_nojit = model(x).detach().cpu()
        time_nojit = time.perf_counter() - t0
    model_nojit = model
    model = jit_compile(model)  # type: ignore
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


def test_issue_89() -> None:
    """Do not fail if 'git' is not installed."""
    model_obj = get_registered_model("breast-tumor-resnet34.tcga-brca")
    d = _get_info_for_save(model_obj)
    assert d
    assert "git" in d["runtime"]
    assert d["runtime"]["git"]
    assert d["runtime"]["git"]["git_remote"]
    assert d["runtime"]["git"]["git_branch"]

    # Test that _get_info_for_save does not fail if git is not found.
    orig_path = os.environ["PATH"]
    try:
        os.environ["PATH"] = ""
        d = _get_info_for_save(model_obj)
        assert d
        assert "git" in d["runtime"]
        assert d["runtime"]["git"] is None
    finally:
        os.environ["PATH"] = orig_path  # reset path


def test_issue_94(tmp_path: Path, tiff_image: Path) -> None:
    """Gracefully handle unreadable slides."""

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
            "--results-dir",
            str(results_dir),
            "--model",
            "breast-tumor-resnet34.tcga-brca",
        ],
    )
    # Important part is that we run through all of the files, despite the unreadble
    # file.
    assert result.exit_code == 0
    assert results_dir.joinpath("model-outputs-csv").joinpath("purple.csv").exists()
    assert not results_dir.joinpath("model-outputs-csv").joinpath("bad.csv").exists()


def test_issue_97(tmp_path: Path, tiff_image: Path) -> None:
    """Write a run_metadata file per run."""

    runner = CliRunner()
    results_dir = tmp_path / "inference"
    result = runner.invoke(
        cli,
        [
            "run",
            "--wsi-dir",
            str(tiff_image.parent),
            "--results-dir",
            str(results_dir),
            "--model",
            "breast-tumor-resnet34.tcga-brca",
        ],
    )
    assert result.exit_code == 0
    metas = list(results_dir.glob("run_metadata_*.json"))
    assert len(metas) == 1

    time.sleep(2)  # make sure some time has passed so the timestamp is different

    # Run again...
    result = runner.invoke(
        cli,
        [
            "run",
            "--wsi-dir",
            str(tiff_image.parent),
            "--results-dir",
            str(results_dir),
            "--model",
            "breast-tumor-resnet34.tcga-brca",
        ],
    )
    assert result.exit_code == 0
    metas = list(results_dir.glob("run_metadata_*.json"))
    assert len(metas) == 2


def test_issue_125(tmp_path: Path) -> None:
    """Test that path in model config can be saved when a pathlib.Path object."""

    w = get_registered_model("breast-tumor-resnet34.tcga-brca")
    w.model_path = Path(w.model_path)  # type: ignore
    info = _get_info_for_save(w)
    with open(tmp_path / "foo.json", "w") as f:
        json.dump(info, f)
