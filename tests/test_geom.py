import geopandas as gpd
from pathlib import Path
import h5py
from pyproj import CRS
from src.rashdf.geom import RasGeomHdf, RasGeomHdfError
from pandas.testing import assert_frame_equal
import pytest
import numpy as np

from . import _create_hdf_with_group_attrs, _gdf_matches_json, _gdf_matches_json_alt

TEST_DATA = Path("./tests/data")
MUNCIE_G05 = TEST_DATA / "ras/Muncie.g05.hdf"
COAL_G01 = TEST_DATA / "ras/Coal.g01.hdf"
BAXTER_P01 = TEST_DATA / "ras_1d/Baxter.p01.hdf"
TEST_JSON = TEST_DATA / "json"
BALD_EAGLE_P18_REF = TEST_DATA / "ras/BaldEagleDamBrk.reflines-refpts.p18.hdf"
LOWER_KANAWHA_P01_IC_POINTS = TEST_DATA / "ras/LowerKanawha.p01.icpoints.hdf"
LOWER_KANAWHA_P01_IC_POINTS_JSON = TEST_JSON / "LowerKanawha.p01.icpoints.geojson"

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


def test_mesh_area_names():
    with RasGeomHdf(MUNCIE_G05) as ghdf:
        assert ghdf.mesh_area_names() == ["2D Interior Area", "Perimeter_NW"]


def test_invalid_mesh_area_names(tmp_path):
    test_hdf = tmp_path / "test.hdf"
    _create_hdf_with_group_attrs(test_hdf, RasGeomHdf.GEOM_PATH, TEST_ATTRS)
    # Test the empty Mesh Area names
    with RasGeomHdf(test_hdf) as ghdf:
        assert ghdf.mesh_area_names() == []


def test_missing_mesh_in_mesh_area(tmp_path):
    test_hdf = tmp_path / "test.hdf"
    mesh_name = "blw-west-fork"
    with h5py.File(test_hdf, "w") as f:
        dtype = np.dtype([("Name", h5py.string_dtype("utf-8"))])
        data = np.array([(mesh_name,)], dtype=dtype)
        f.create_dataset(f"{RasGeomHdf.FLOW_AREA_2D_PATH}/Attributes", data=data)

    ras_hdf = RasGeomHdf(test_hdf)
    expected_error_message = f"Data for mesh '{mesh_name}' not found."
    with pytest.raises(RasGeomHdfError, match=expected_error_message):
        ras_hdf.mesh_areas()


def test_mesh_areas():
    mesh_areas_json = TEST_JSON / "mesh_areas.json"
    with RasGeomHdf(MUNCIE_G05) as ghdf:
        assert _gdf_matches_json(ghdf.mesh_areas(), mesh_areas_json)


def test_invalid_mesh_areas(tmp_path):
    test_hdf = tmp_path / "test.hdf"
    _create_hdf_with_group_attrs(test_hdf, RasGeomHdf.GEOM_PATH, TEST_ATTRS)
    # Test the empty Mesh Areas
    with RasGeomHdf(test_hdf) as ghdf:
        assert ghdf.mesh_areas().empty


def test_mesh_cell_faces():
    mesh_cell_faces_json = TEST_JSON / "mesh_cell_faces.json"
    with RasGeomHdf(MUNCIE_G05) as ghdf:
        assert _gdf_matches_json(ghdf.mesh_cell_faces(), mesh_cell_faces_json)


def test_invalid_mesh_faces(tmp_path):
    test_hdf = tmp_path / "test.hdf"
    _create_hdf_with_group_attrs(test_hdf, RasGeomHdf.GEOM_PATH, TEST_ATTRS)
    # Test the empty Mesh Faces
    with RasGeomHdf(test_hdf) as ghdf:
        assert ghdf.mesh_cell_faces().empty


def test_mesh_cell_points():
    mesh_cell_points_json = TEST_JSON / "mesh_cell_points.json"
    with RasGeomHdf(MUNCIE_G05) as ghdf:
        assert _gdf_matches_json(ghdf.mesh_cell_points(), mesh_cell_points_json)


def test_invalid_mesh_cell_points(tmp_path):
    test_hdf = tmp_path / "test.hdf"
    _create_hdf_with_group_attrs(test_hdf, RasGeomHdf.GEOM_PATH, TEST_ATTRS)
    # Test the empty Mesh Cell Points
    with RasGeomHdf(test_hdf) as ghdf:
        assert ghdf.mesh_cell_points().empty


def test_mesh_cell_polygons():
    mesh_cell_polygons_json = TEST_JSON / "mesh_cell_polygons.json"
    with RasGeomHdf(MUNCIE_G05) as ghdf:
        assert _gdf_matches_json(ghdf.mesh_cell_polygons(), mesh_cell_polygons_json)


def test_invalid_mesh_cell_polygons(tmp_path):
    # Create a dummy HDF file
    test_hdf = tmp_path / "test.hdf"
    _create_hdf_with_group_attrs(test_hdf, RasGeomHdf.GEOM_PATH, TEST_ATTRS)
    # Test the empty Mesh Cell Polygons
    with RasGeomHdf(test_hdf) as ghdf:
        assert ghdf.mesh_cell_polygons().empty


def test_mesh_cell_polygons_coal():
    """Test with the mesh from the Coal River model.

    The Jan 2024 Coal River model from the Kanawha FFRD pilot project
    contains some topologically incorrect polygons in the 2D mesh;
    some of the mesh cell faces overlap.

    See: https://github.com/fema-ffrd/rashdf/issues/31
    """
    coal_bad_polygons_json = TEST_JSON / "coal-bad-mesh-cell-polygons.json"
    with RasGeomHdf(COAL_G01) as geom_hdf:
        gdf = geom_hdf.mesh_cell_polygons()
        gdf_bad_polygons = gdf[gdf["cell_id"].isin([8561, 11791, 17529])].to_crs(
            "EPSG:4326"
        )  # reproject because the Kanawha FFRD pilot project uses a custom CRS
        assert _gdf_matches_json(gdf_bad_polygons, coal_bad_polygons_json)


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


def test_invalid_get_geom_2d_flow_area_attrs(tmp_path):
    test_hdf = tmp_path / "test.hdf"
    _create_hdf_with_group_attrs(test_hdf, RasGeomHdf.GEOM_PATH, TEST_ATTRS)
    ras_hdf = RasGeomHdf(test_hdf)

    with pytest.raises(
        AttributeError,
        match=f"Unable to get 2D Flow Area; {RasGeomHdf.FLOW_AREA_2D_PATH} group not found in HDF5 file.",
    ):
        ras_hdf.get_geom_2d_flow_area_attrs()


def test_structs():
    structs_json = TEST_JSON / "structures.json"
    with RasGeomHdf(MUNCIE_G05) as ghdf:
        assert _gdf_matches_json(ghdf.structures(datetime_to_str=True), structs_json)


def test_reference_lines_names():
    with RasGeomHdf(BALD_EAGLE_P18_REF) as geom_hdf:
        assert geom_hdf.reference_lines_names() == {
            "BaldEagleCr": [
                "Reference Line 1",
                "Reference Line 2",
                "Reference Line 3",
                "Reference Line 4",
            ]
        }
        assert geom_hdf.reference_lines_names("Upper 2D Area") == []


def test_reference_points_names():
    with RasGeomHdf(BALD_EAGLE_P18_REF) as geom_hdf:
        assert geom_hdf.reference_points_names() == {
            "Upper 2D Area": [
                "Reference Point 1",
            ],
            "BaldEagleCr": [
                "Reference Point 2",
                "Reference Point 3",
            ],
        }
        assert geom_hdf.reference_points_names("Upper 2D Area") == [
            "Reference Point 1",
        ]


def test_structs_not_found():
    with RasGeomHdf(COAL_G01) as ghdf:
        assert ghdf.structures().empty


def test_cross_sections():
    cross_section_json = TEST_JSON / "cross_sections.json"
    with RasGeomHdf(BAXTER_P01) as ghdf:
        assert _gdf_matches_json_alt(
            ghdf.cross_sections(datetime_to_str=True), cross_section_json
        )


def test_cross_sections_not_found():
    with RasGeomHdf(COAL_G01) as ghdf:
        assert ghdf.cross_sections().empty


def test_river_reaches():
    river_reaches_json = TEST_JSON / "river_reaches.json"
    with RasGeomHdf(BAXTER_P01) as ghdf:
        assert _gdf_matches_json_alt(ghdf.river_reaches(), river_reaches_json)


def test_river_reaches_not_found():
    with RasGeomHdf(COAL_G01) as ghdf:
        assert ghdf.river_reaches().empty


def test_cross_sections_elevations():
    xs_elevs_json = TEST_JSON / "xs_elevations.json"
    with RasGeomHdf(BAXTER_P01) as ghdf:
        assert _gdf_matches_json_alt(ghdf.cross_sections_elevations(), xs_elevs_json)


def test_cross_sections_elevations_not_found():
    with RasGeomHdf(COAL_G01) as ghdf:
        assert ghdf.cross_sections_elevations().empty


def test_ic_points():
    with RasGeomHdf(LOWER_KANAWHA_P01_IC_POINTS) as ghdf:
        gdf_ic_points = ghdf.ic_points()
        valid_gdf = gpd.read_file(
            LOWER_KANAWHA_P01_IC_POINTS_JSON,
            crs=ghdf.projection(),
        )
        assert_frame_equal(
            gdf_ic_points,
            valid_gdf,
            check_dtype=False,
        )
