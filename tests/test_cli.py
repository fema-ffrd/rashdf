from src.cli import parse_args, export, docstring_to_help, fiona_supported_drivers

import geopandas as gpd
from pyproj import CRS

from pathlib import Path

TEST_DATA = Path("./tests/data")
MUNCIE_G05 = TEST_DATA / "ras/Muncie.g05.hdf"


def test_docstring_to_help():
    docstring = """This is a test docstring.
    This is not part of the help message.
    """
    assert docstring_to_help(docstring) == "This is a test docstring."

    docstring = """Return the something or other.
    Blah blah blah."""
    assert docstring_to_help(docstring) == "Export the something or other."


def test_fiona_supported_drivers():
    drivers = fiona_supported_drivers()
    assert "ESRI Shapefile" in drivers
    assert "GeoJSON" in drivers
    assert "GPKG" in drivers


def test_parse_args():
    args = parse_args(["mesh_areas", "test.hdf", "test.json"])
    assert args.func == "mesh_areas"
    assert args.hdf_file == "test.hdf"
    assert args.output_file == "test.json"
    assert args.to_crs is None
    assert not args.parquet
    assert not args.feather
    assert not args.json
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
    assert not args.json
    assert args.kwargs == '{"compression": "gzip"}'

    args = parse_args(["--fiona-drivers"])
    assert args.fiona_drivers


def test_export(tmp_path: Path):
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
