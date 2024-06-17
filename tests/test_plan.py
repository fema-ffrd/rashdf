from src.rashdf.plan import RasPlanHdf, SummaryOutputVar, RasPlanHdfError

from pathlib import Path

import pytest

from . import _create_hdf_with_group_attrs, _gdf_matches_json, get_sha1_hash

TEST_DATA = Path("./tests/data")
TEST_JSON = TEST_DATA / "json"
TEST_ATTRS = {"test_attribute1": "test_str1", "test_attribute2": 500}
BALD_EAGLE_P18 = TEST_DATA / "ras/BaldEagleDamBrk.p18.hdf"
MUNCIE_G05 = TEST_DATA / "ras/Muncie.g05.hdf"


def test_get_plan_info_attrs(tmp_path):
    test_hdf = tmp_path / "test.hdf"
    _create_hdf_with_group_attrs(test_hdf, RasPlanHdf.PLAN_INFO_PATH, TEST_ATTRS)
    ras_plan_hdf = RasPlanHdf(test_hdf)
    assert ras_plan_hdf.get_plan_info_attrs() == TEST_ATTRS


def test_get_plan_param_attrs(tmp_path):
    test_hdf = tmp_path / "test.hdf"
    _create_hdf_with_group_attrs(test_hdf, RasPlanHdf.PLAN_PARAMS_PATH, TEST_ATTRS)
    ras_plan_hdf = RasPlanHdf(test_hdf)
    assert ras_plan_hdf.get_plan_param_attrs() == TEST_ATTRS


def test_get_meteorology_precip_attrs(tmp_path):
    test_hdf = tmp_path / "test.hdf"
    _create_hdf_with_group_attrs(test_hdf, RasPlanHdf.PRECIP_PATH, TEST_ATTRS)
    ras_plan_hdf = RasPlanHdf(test_hdf)
    assert ras_plan_hdf.get_meteorology_precip_attrs() == TEST_ATTRS


def test_get_results_unsteady_attrs(tmp_path):
    test_hdf = tmp_path / "test.hdf"
    _create_hdf_with_group_attrs(test_hdf, RasPlanHdf.RESULTS_UNSTEADY_PATH, TEST_ATTRS)
    ras_plan_hdf = RasPlanHdf(test_hdf)
    assert ras_plan_hdf.get_results_unsteady_attrs() == TEST_ATTRS


def test_get_results_unsteady_summary_attrs(tmp_path):
    test_hdf = tmp_path / "test.hdf"
    _create_hdf_with_group_attrs(
        test_hdf, RasPlanHdf.RESULTS_UNSTEADY_SUMMARY_PATH, TEST_ATTRS
    )
    ras_plan_hdf = RasPlanHdf(test_hdf)
    assert ras_plan_hdf.get_results_unsteady_summary_attrs() == TEST_ATTRS


def test_get_results_volume_accounting_attrs(tmp_path):
    test_hdf = tmp_path / "test.hdf"
    _create_hdf_with_group_attrs(
        test_hdf, RasPlanHdf.VOLUME_ACCOUNTING_PATH, TEST_ATTRS
    )
    ras_plan_hdf = RasPlanHdf(test_hdf)
    assert ras_plan_hdf.get_results_volume_accounting_attrs() == TEST_ATTRS


def test__mesh_summary_output_group_null():
    """Test that an exception is raised if the specified mesh summary output data doesn't exist."""
    with RasPlanHdf(BALD_EAGLE_P18) as plan_hdf:
        with pytest.raises(RasPlanHdfError):
            plan_hdf._mesh_summary_output_group(
                "Nonexistent Mesh", SummaryOutputVar.MINIMUM_WATER_SURFACE
            )


def test_mesh_cell_points():
    # Test that the mesh cell points are returned with no other columns if
    # "include_output" is False.
    mesh_cell_points_json = TEST_JSON / "mesh_cell_points.json"
    with RasPlanHdf(MUNCIE_G05) as plan_hdf:
        assert _gdf_matches_json(
            plan_hdf.mesh_cell_points(include_output=False), mesh_cell_points_json
        )


def test_include_output_type_error():
    with RasPlanHdf(BALD_EAGLE_P18) as plan_hdf:
        with pytest.raises(ValueError):
            plan_hdf.mesh_cell_points(include_output="This should be a list instead")


def test_cells_or_values_error():
    """Test that an exception is raised if the cells_or_faces parameter is not 'cells' or 'faces'."""
    with RasPlanHdf(BALD_EAGLE_P18) as plan_hdf:
        with pytest.raises(ValueError):
            plan_hdf._mesh_summary_outputs_gdf(
                "mesh_cell_points", cells_or_faces="neither"
            )


def test_mesh_cell_points_all_outputs_columns():
    """Test that all summary output columns are returned if include_output is True."""
    plan_hdf = RasPlanHdf(BALD_EAGLE_P18)
    gdf = plan_hdf.mesh_cell_points(include_output=True)
    expected_columns = [
        "mesh_name",
        "cell_id",
        "geometry",
        "max_iter",
        "last_iter",
        "max_ws_err",
        "max_ws_err_time",
        "max_ws",
        "max_ws_time",
        "min_ws",
        "min_ws_time",
    ]
    assert list(gdf.columns) == expected_columns


def test_mesh_cell_points_with_output(tmp_path):
    plan_hdf = RasPlanHdf(BALD_EAGLE_P18)
    gdf = plan_hdf.mesh_cell_points(
        datetime_to_str=True,
        include_output=[
            SummaryOutputVar.CELL_MAXIMUM_WATER_SURFACE_ERROR,
            "Maximum Water Surface",
            "Minimum Water Surface",
        ],
    )
    temp_points = tmp_path / "temp-bald-eagle-mesh-cell-points.geojson"
    gdf = gdf.to_crs(4326)
    gdf.to_file(temp_points)
    valid = get_sha1_hash(TEST_JSON / "bald-eagle-mesh-cell-points.geojson")
    test = get_sha1_hash(temp_points)
    assert valid == test


def test_mesh_cell_polygons_with_output(tmp_path):
    plan_hdf = RasPlanHdf(BALD_EAGLE_P18)
    gdf = plan_hdf.mesh_cell_polygons(
        datetime_to_str=True,
        include_output=[
            SummaryOutputVar.CELL_MAXIMUM_WATER_SURFACE_ERROR,
            "Maximum Water Surface",
            "Minimum Water Surface",
        ],
    )
    temp_polygons = tmp_path / "temp-bald-eagle-mesh-cell-polygons.geojson"
    gdf.to_crs(4326).to_file(temp_polygons)
    valid = get_sha1_hash(TEST_JSON / "bald-eagle-mesh-cell-polygons.geojson")
    test = get_sha1_hash(temp_polygons)
    assert valid == test


def test_mesh_cell_faces_with_output(tmp_path):
    plan_hdf = RasPlanHdf(BALD_EAGLE_P18)
    gdf = plan_hdf.mesh_cell_faces(
        datetime_to_str=True,
        include_output=[
            "Minimum Face Velocity",
            SummaryOutputVar.MAXIMUM_FACE_VELOCITY,
        ],
    )
    temp_faces = tmp_path / "temp-bald-eagle-mesh-cell-faces.geojson"
    gdf.to_crs(4326).to_file(temp_faces)
    valid = get_sha1_hash(TEST_JSON / "bald-eagle-mesh-cell-faces.geojson")
    test = get_sha1_hash(temp_faces)
    assert valid == test
