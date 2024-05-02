from src.rashdf import RasPlanHdf

from . import _create_hdf_with_group_attrs

TEST_ATTRS = {"test_attribute1": "test_str1", "test_attribute2": 500}


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
