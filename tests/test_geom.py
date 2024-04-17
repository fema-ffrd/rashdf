from src.rashdf import RasGeomHdf

import h5py
from pyproj import CRS
from pathlib import Path

TEST_DATA = Path("./tests/data")

def test_projection(tmp_path):
    wkt = 'PROJCS["Albers_Conic_Equal_Area",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",-96.0],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["latitude_of_origin",37.5],UNIT["Foot_US",0.3048006096012192]]'
    # Create a dummy HDF file
    with h5py.File(tmp_path / "test.hdf", "w") as f:
        f.attrs["Projection"] = wkt.encode("utf-8")
    # Open the HDF file
    ras_hdf = RasGeomHdf(tmp_path / "test.hdf")
    # Test the projection
    assert ras_hdf.projection() == CRS.from_wkt(wkt)

def test_mesh_area_names():
    geom = TEST_DATA / "ras/Muncie.g05.hdf"
    with RasGeomHdf(geom) as ghdf:
        assert ghdf.mesh_area_names() == ["2D Interior Area", "Perimeter_NW"]

def test_mesh_areas():
    geom = TEST_DATA / "ras/Muncie.g05.hdf"
    with RasGeomHdf(geom) as ghdf:
        with open(TEST_DATA / "json/mesh_areas.json") as json:
            assert ghdf.mesh_areas().to_json() == json.read()

def test_mesh_cell_faces():
    geom = TEST_DATA / "ras/Muncie.g05.hdf"
    with RasGeomHdf(geom) as ghdf:
        with open(TEST_DATA / "json/mesh_cell_faces.json") as json:
            assert ghdf.mesh_cell_faces().to_json() == json.read()

def test_mesh_cell_points():
    geom = TEST_DATA / "ras/Muncie.g05.hdf"
    with RasGeomHdf(geom) as ghdf:
        with open(TEST_DATA / "json/mesh_cell_points.json") as json:
            assert ghdf.mesh_cell_points().to_json() == json.read()

def test_mesh_cell_polygons():
    geom = TEST_DATA / "ras/Muncie.g05.hdf"
    with RasGeomHdf(geom) as ghdf:
        with open(TEST_DATA / "json/mesh_cell_polygons.json") as json:
            assert ghdf.mesh_cell_polygons().to_json() == json.read()