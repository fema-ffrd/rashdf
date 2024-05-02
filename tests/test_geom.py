from src.rashdf import RasGeomHdf

import h5py
from geopandas import GeoDataFrame
from pyproj import CRS
from pathlib import Path

from . import _create_hdf_with_group_attrs

TEST_DATA = Path("./tests/data")
MUNCIE_G05 = TEST_DATA / "ras/Muncie.g05.hdf"
TEST_JSON = TEST_DATA / "json"

TEST_ATTRS = {"test_attribute1": "test_str1", "test_attribute2": 500}


def test_projection(tmp_path):
    wkt = 'PROJCS["Albers_Conic_Equal_Area",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",-96.0],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["latitude_of_origin",37.5],UNIT["Foot_US",0.3048006096012192]]'
    # Create a dummy HDF file
    with h5py.File(tmp_path / "test.hdf", "w") as f:
        f.attrs["Projection"] = wkt.encode("utf-8")
    # Open the HDF file
    ras_hdf = RasGeomHdf(tmp_path / "test.hdf")
    # Test the projection
    assert ras_hdf.projection() == CRS.from_wkt(wkt)


def _gdf_matches_json(gdf: GeoDataFrame, json_file: Path) -> bool:
    with open(json_file) as j:
        return gdf.to_json() == j.read()


def test_mesh_area_names():
    with RasGeomHdf(MUNCIE_G05) as ghdf:
        assert ghdf.mesh_area_names() == ["2D Interior Area", "Perimeter_NW"]


def test_mesh_areas():
    mesh_areas_json = TEST_JSON / "mesh_areas.json"
    with RasGeomHdf(MUNCIE_G05) as ghdf:
        assert _gdf_matches_json(ghdf.mesh_areas(), mesh_areas_json)


def test_mesh_cell_faces():
    mesh_cell_faces_json = TEST_JSON / "mesh_cell_faces.json"
    with RasGeomHdf(MUNCIE_G05) as ghdf:
        assert _gdf_matches_json(ghdf.mesh_cell_faces(), mesh_cell_faces_json)


def test_mesh_cell_points():
    mesh_cell_points_json = TEST_JSON / "mesh_cell_points.json"
    with RasGeomHdf(MUNCIE_G05) as ghdf:
        assert _gdf_matches_json(ghdf.mesh_cell_points(), mesh_cell_points_json)


def test_mesh_cell_polygons():
    mesh_cell_polygons_json = TEST_JSON / "mesh_cell_polygons.json"
    with RasGeomHdf(MUNCIE_G05) as ghdf:
        assert _gdf_matches_json(ghdf.mesh_cell_polygons(), mesh_cell_polygons_json)


def test_bc_lines():
    bc_lines_json = TEST_JSON / "bc_lines.json"
    with RasGeomHdf(MUNCIE_G05) as ghdf:
        assert _gdf_matches_json(ghdf.bc_lines(), bc_lines_json)


def test_breaklines():
    breaklines_json = TEST_JSON / "breaklines.json"
    with RasGeomHdf(MUNCIE_G05) as ghdf:
        assert _gdf_matches_json(ghdf.breaklines(), breaklines_json)


def test_refinement_regions():
    rr_json = TEST_JSON / "refinement_regions.json"
    with RasGeomHdf(MUNCIE_G05) as ghdf:
        assert _gdf_matches_json(ghdf.refinement_regions(), rr_json)


def test_get_geom_attrs(tmp_path):
    test_hdf = tmp_path / "test.hdf"
    _create_hdf_with_group_attrs(test_hdf, RasGeomHdf.GEOM_PATH, TEST_ATTRS)
    ras_hdf = RasGeomHdf(test_hdf)
    assert ras_hdf.get_geom_attrs() == TEST_ATTRS


def test_get_geom_structures_attrs(tmp_path):
    test_hdf = tmp_path / "test.hdf"
    _create_hdf_with_group_attrs(test_hdf, RasGeomHdf.GEOM_STRUCTURES_PATH, TEST_ATTRS)
    ras_hdf = RasGeomHdf(test_hdf)
    assert ras_hdf.get_geom_structures_attrs() == TEST_ATTRS


def test_get_geom_2d_flow_area_attrs(tmp_path):
    test_hdf = tmp_path / "test.hdf"
    _create_hdf_with_group_attrs(
        test_hdf, f"{RasGeomHdf.FLOW_AREA_2D_PATH}/group", TEST_ATTRS
    )
    ras_hdf = RasGeomHdf(test_hdf)
    assert ras_hdf.get_geom_2d_flow_area_attrs() == TEST_ATTRS
