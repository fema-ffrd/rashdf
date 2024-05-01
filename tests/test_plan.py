from src.rashdf import RasPlanHdf

import h5py

attrs_to_set = {"test_attribute1": "test_str1", "test_attribute2": 500}


def test_get_plan_info_attrs(tmp_path):
    with h5py.File(tmp_path / "test.hdf", "w") as f:
        group = f.create_group(RasPlanHdf.PLAN_INFO_PATH)
        for key, value in attrs_to_set.items():
            group.attrs[key] = value

    ras_plan_hdf = RasPlanHdf(tmp_path / "test.hdf")

    assert ras_plan_hdf.get_plan_info_attrs() == attrs_to_set


def test_get_plan_param_attrs(tmp_path):
    with h5py.File(tmp_path / "test.hdf", "w") as f:
        group = f.create_group(RasPlanHdf.PLAN_PARAMS_PATH)
        for key, value in attrs_to_set.items():
            group.attrs[key] = value

    ras_plan_hdf = RasPlanHdf(tmp_path / "test.hdf")

    assert ras_plan_hdf.get_plan_param_attrs() == attrs_to_set


def test_get_meteorology_precip_attrs(tmp_path):
    with h5py.File(tmp_path / "test.hdf", "w") as f:
        group = f.create_group(RasPlanHdf.PRECIP_PATH)
        for key, value in attrs_to_set.items():
            group.attrs[key] = value

    ras_plan_hdf = RasPlanHdf(tmp_path / "test.hdf")

    assert ras_plan_hdf.get_meteorology_precip_attrs() == attrs_to_set


def test_get_results_unsteady_attrs(tmp_path):
    with h5py.File(tmp_path / "test.hdf", "w") as f:
        group = f.create_group(RasPlanHdf.RESULTS_UNSTEADY_PATH)
        for key, value in attrs_to_set.items():
            group.attrs[key] = value

    ras_plan_hdf = RasPlanHdf(tmp_path / "test.hdf")

    assert ras_plan_hdf.get_results_unsteady_attrs() == attrs_to_set


def test_get_results_unsteady_summary_attrs(tmp_path):
    with h5py.File(tmp_path / "test.hdf", "w") as f:
        group = f.create_group(RasPlanHdf.RESULTS_UNSTEADY_SUMMARY_PATH)
        for key, value in attrs_to_set.items():
            group.attrs[key] = value

    ras_plan_hdf = RasPlanHdf(tmp_path / "test.hdf")

    assert ras_plan_hdf.get_results_unsteady_summary_attrs() == attrs_to_set


def test_get_results_volume_accounting_attrs(tmp_path):
    with h5py.File(tmp_path / "test.hdf", "w") as f:
        group = f.create_group(RasPlanHdf.VOLUME_ACCOUNTING_PATH)
        for key, value in attrs_to_set.items():
            group.attrs[key] = value

    ras_plan_hdf = RasPlanHdf(tmp_path / "test.hdf")

    assert ras_plan_hdf.get_results_volume_accounting_attrs() == attrs_to_set
