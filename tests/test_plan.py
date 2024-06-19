from src.rashdf.plan import (
    RasPlanHdf,
    SummaryOutputVar,
    RasPlanHdfError,
    TimeSeriesOutputVar,
)

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from . import (
    _create_hdf_with_group_attrs,
    _gdf_matches_json,
    get_sha1_hash,
    _gdf_matches_json_alt,
)

TEST_DATA = Path("./tests/data")
TEST_JSON = TEST_DATA / "json"
TEST_CSV = TEST_DATA / "csv"
TEST_ATTRS = {"test_attribute1": "test_str1", "test_attribute2": 500}
BALD_EAGLE_P18 = TEST_DATA / "ras/BaldEagleDamBrk.p18.hdf"
BALD_EAGLE_P18_TIMESERIES = TEST_DATA / "ras/BaldEagleDamBrk.p18.timeseries.hdf"
MUNCIE_G05 = TEST_DATA / "ras/Muncie.g05.hdf"
COAL_G01 = TEST_DATA / "ras/Coal.g01.hdf"
BAXTER_P01 = TEST_DATA / "ras_1d/Baxter.p01.hdf"
FLODENCR_P01 = TEST_DATA / "ras_1d/FLODENCR.p01.hdf"


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


def test_mesh_timeseries_output():
    with RasPlanHdf(BALD_EAGLE_P18_TIMESERIES) as plan_hdf:
        with pytest.raises(ValueError):
            plan_hdf.mesh_timeseries_output(
                "Fake Mesh", TimeSeriesOutputVar.WATER_SURFACE
            )
        with pytest.raises(ValueError):
            plan_hdf.mesh_timeseries_output("BaldEagleCr", "Fake Variable")


def test_mesh_timeseries_output_cells():
    with RasPlanHdf(BALD_EAGLE_P18_TIMESERIES) as plan_hdf:
        ds = plan_hdf.mesh_timeseries_output_cells("BaldEagleCr")
        assert "time" in ds.coords
        assert "cell_id" in ds.coords
        assert "Water Surface" in ds.variables
        ws = ds["Water Surface"]
        assert ws.shape == (37, 3359)
        assert ws.attrs["units"] == "ft"
        assert ws.attrs["mesh_name"] == "BaldEagleCr"
        df = ws.sel(cell_id=123).to_dataframe()
        valid_df = pd.read_csv(
            TEST_CSV / "BaldEagleDamBrk.BaldEagleCr.ws.123.csv",
            index_col="time",
            parse_dates=True,
            dtype={"Water Surface": np.float32},
        )
        assert_frame_equal(df, valid_df)

        ds = plan_hdf.mesh_timeseries_output_cells("Upper 2D Area")
        assert "time" in ds.coords
        assert "cell_id" in ds.coords
        assert "Water Surface" in ds.variables
        ws = ds["Water Surface"]
        assert ws.shape == (37, 1066)
        assert ws.attrs["units"] == "ft"
        assert ws.attrs["mesh_name"] == "Upper 2D Area"
        df = ws.sel(cell_id=456).to_dataframe()
        valid_df = pd.read_csv(
            TEST_CSV / "BaldEagleDamBrk.Upper2DArea.ws.456.csv",
            index_col="time",
            parse_dates=True,
            dtype={"Water Surface": np.float32},
        )
        assert_frame_equal(df, valid_df)


def test_mesh_timeseries_output_faces():
    with RasPlanHdf(BALD_EAGLE_P18_TIMESERIES) as plan_hdf:
        ds = plan_hdf.mesh_timeseries_output_faces("BaldEagleCr")
        assert "time" in ds.coords
        assert "face_id" in ds.coords
        assert "Face Velocity" in ds.variables
        v = ds["Face Velocity"]
        assert v.shape == (37, 7295)
        assert v.attrs["units"] == "ft/s"
        assert v.attrs["mesh_name"] == "BaldEagleCr"
        df = v.sel(face_id=678).to_dataframe()
        valid_df = pd.read_csv(
            TEST_CSV / "BaldEagleDamBrk.BaldEagleCr.v.678.csv",
            index_col="time",
            parse_dates=True,
            dtype={"Face Velocity": np.float32},
        )
        assert_frame_equal(df, valid_df)

        ds = plan_hdf.mesh_timeseries_output_faces("Upper 2D Area")
        assert "time" in ds.coords
        assert "face_id" in ds.coords
        assert "Face Velocity" in ds.variables
        v = ds["Face Velocity"]
        assert v.shape == (37, 2286)
        assert v.attrs["units"] == "ft/s"
        assert v.attrs["mesh_name"] == "Upper 2D Area"
        df = v.sel(face_id=567).to_dataframe()
        valid_df = pd.read_csv(
            TEST_CSV / "BaldEagleDamBrk.Upper2DArea.v.567.csv",
            index_col="time",
            parse_dates=True,
            dtype={"Face Velocity": np.float32},
        )
        assert_frame_equal(df, valid_df)


def test_cross_sections_additional_velocity_total():
    xs_velocity_json = TEST_JSON / "xs_velocity.json"
    with RasPlanHdf(BAXTER_P01) as phdf:
        assert _gdf_matches_json_alt(
            phdf.cross_sections_additional_velocity_total(), xs_velocity_json
        )


def test_cross_sections_additional_velocity_total_not_found():
    with RasPlanHdf(COAL_G01) as phdf:
        assert (phdf.cross_sections_additional_velocity_total(), None)


def test_cross_sections_additional_area_total():
    xs_area_json = TEST_JSON / "xs_area.json"
    with RasPlanHdf(BAXTER_P01) as phdf:
        assert _gdf_matches_json_alt(
            phdf.cross_sections_additional_area_total(), xs_area_json
        )


def test_cross_sections_additional_area_total_not_found():
    with RasPlanHdf(COAL_G01) as phdf:
        assert (phdf.cross_sections_additional_area_total(), None)


def test_steady_flow_names():
    with RasPlanHdf(BAXTER_P01) as phdf:
        assert phdf.steady_flow_names() == ["Big"]


def test_steady_flow_names_not_found():
    with RasPlanHdf(COAL_G01) as phdf:
        assert (phdf.steady_flow_names(), None)


def test_cross_sections_wsel():
    xs_wsel_json = TEST_JSON / "xs_wsel.json"
    with RasPlanHdf(BAXTER_P01) as phdf:
        assert _gdf_matches_json_alt(phdf.cross_sections_wsel(), xs_wsel_json)


def test_cross_sections_wsel_not_found():
    with RasPlanHdf(COAL_G01) as phdf:
        assert (phdf.cross_sections_wsel(), None)


def test_cross_sections_additional_enc_station_right():
    xs_enc_station_right_json = TEST_JSON / "xs_enc_station_right.json"
    with RasPlanHdf(FLODENCR_P01) as phdf:
        assert _gdf_matches_json_alt(
            phdf.cross_sections_additional_enc_station_right(),
            xs_enc_station_right_json,
        )


def test_cross_sections_additional_enc_station_right_not_found():
    with RasPlanHdf(COAL_G01) as phdf:
        assert (phdf.cross_sections_additional_enc_station_right(), None)


def test_cross_sections_additional_enc_station_left():
    xs_enc_station_left_json = TEST_JSON / "xs_enc_station_left.json"
    with RasPlanHdf(FLODENCR_P01) as phdf:
        assert _gdf_matches_json_alt(
            phdf.cross_sections_additional_enc_station_left(), xs_enc_station_left_json
        )


def test_cross_sections_additional_enc_station_left_not_found():
    with RasPlanHdf(COAL_G01) as phdf:
        assert (phdf.cross_sections_additional_enc_station_left(), None)


def test_cross_sections_flow():
    xs_flow_json = TEST_JSON / "xs_flow.json"
    with RasPlanHdf(BAXTER_P01) as phdf:
        assert _gdf_matches_json_alt(phdf.cross_sections_flow(), xs_flow_json)


def test_cross_sections_energy_grade():
    xs_energy_grade_json = TEST_JSON / "xs_energy_grade.json"
    with RasPlanHdf(BAXTER_P01) as phdf:
        assert _gdf_matches_json_alt(
            phdf.cross_sections_energy_grade(), xs_energy_grade_json
        )
