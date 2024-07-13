from src.cli import (
    parse_args,
    export,
    docstring_to_help,
    fiona_supported_drivers,
    pyogrio_supported_drivers,
)

import geopandas as gpd
from pyproj import CRS
import pytest

import builtins
import json
from pathlib import Path

TEST_DATA = Path("./tests/data")
MUNCIE_G05 = TEST_DATA / "ras/Muncie.g05.hdf"
BALD_EAGLE_P18 = TEST_DATA / "ras/BaldEagleDamBrk.p18.hdf"


def test_docstring_to_help():
    docstring = """This is a test docstring.
    This is not part of the help message.
    """
    assert docstring_to_help(docstring) == "This is a test docstring."

    docstring = """Return the something or other.
    Blah blah blah."""
    assert docstring_to_help(docstring) == "Export the something or other."

    docstring = None
    assert docstring_to_help(docstring) == ""


def test_fiona_supported_drivers():
    drivers = fiona_supported_drivers()
    assert "ESRI Shapefile" in drivers
    assert "GeoJSON" in drivers
    assert "GPKG" in drivers
    assert "MBTiles" not in drivers


def test_pyogrio_supported_drivers():
    drivers = pyogrio_supported_drivers()
    assert "ESRI Shapefile" in drivers
    assert "GeoJSON" in drivers
    assert "GPKG" in drivers
    assert "MBTiles" in drivers


def test_parse_args():
    args = parse_args(["structures", "test.hdf"])
    assert args.func == "structures"
    assert args.hdf_file == "test.hdf"
    assert args.output_file is None
    assert args.to_crs is None
    assert not args.parquet
    assert not args.feather
    assert args.kwargs is None

    args = parse_args(["mesh_areas", "test.hdf", "test.json"])
    assert args.func == "mesh_areas"
    assert args.hdf_file == "test.hdf"
    assert args.output_file == "test.json"
    assert args.to_crs is None
    assert not args.parquet
    assert not args.feather
    assert args.kwargs is None

    args = parse_args(
        [
            "mesh_areas",
            "test.hdf",
            "test.json",
            "--to-crs",
            "EPSG:4326",
            "--parquet",
            "--kwargs",
            '{"compression": "gzip"}',
        ]
    )
    assert args.func == "mesh_areas"
    assert args.hdf_file == "test.hdf"
    assert args.output_file == "test.json"
    assert args.to_crs == "EPSG:4326"
    assert args.parquet
    assert not args.feather
    assert args.kwargs == '{"compression": "gzip"}'

    args = parse_args(["--fiona-drivers"])
    assert args.fiona_drivers


def test_export(tmp_path: Path):
    args = parse_args(["structures", str(MUNCIE_G05)])
    exported = json.loads(export(args))
    gdf = gpd.GeoDataFrame.from_features(exported)
    assert len(gdf) == 3
    assert gdf["Last Edited"].to_list() == [
        "2024-04-15T15:21:34",
        "2024-04-15T15:21:48",
        "2024-04-15T15:26:15",
    ]

    test_json_path = tmp_path / "test.json"
    args = parse_args(["mesh_areas", str(MUNCIE_G05), str(test_json_path)])
    export(args)
    gdf = gpd.read_file(test_json_path)
    assert len(gdf) == 2

    test_parquet_path = tmp_path / "test.parquet"
    args = parse_args(
        [
            "mesh_cell_points",
            str(MUNCIE_G05),
            str(test_parquet_path),
            "--parquet",
            "--to-crs",
            "EPSG:4326",
        ]
    )
    export(args)
    gdf = gpd.read_parquet(test_parquet_path)
    assert len(gdf) == 5790
    assert gdf.crs == CRS.from_epsg(4326)


def test_export_plan_hdf():
    args = parse_args(["mesh_cell_points", str(BALD_EAGLE_P18)])
    exported = json.loads(export(args))
    gdf = gpd.GeoDataFrame.from_features(exported)
    assert len(gdf) == 4425


def test_fiona_missing(monkeypatch):
    # Test behavior when fiona isn't installed

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fiona":
            raise ImportError("No module named 'fiona'")
        return real_import(name, globals, locals, fromlist, level)

    real_import = builtins.__import__

    # Replace the built-in __import__ function with our mock
    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Verify that the --fiona-drivers argument is not available
    # when fiona is not installed
    args = parse_args(["structures", "fake_file.hdf"])
    assert not hasattr(args, "fiona_drivers")


def test_print_pyogrio_supported_drivers(capfd):
    export(parse_args(["--pyogrio-drivers"]))
    captured = capfd.readouterr()
    assert "ESRI Shapefile" in captured.out
    assert "GeoJSON" in captured.out
    assert "GPKG" in captured.out
    assert "MBTiles" in captured.out


def test_print_fiona_supported_drivers(capfd):
    export(parse_args(["--fiona-drivers"]))
    captured = capfd.readouterr()
    assert "ESRI Shapefile" in captured.out
    assert "GeoJSON" in captured.out
    assert "GPKG" in captured.out
    assert "MBTiles" not in captured.out
