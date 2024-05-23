from src.rashdf import utils

import numpy as np
import pytest

from datetime import datetime, timedelta


def test_convert_ras_hdf_value():
    assert utils.convert_ras_hdf_value(b"True") is True
    assert utils.convert_ras_hdf_value(b"False") is False
    assert utils.convert_ras_hdf_value(np.float32(1.23)) == pytest.approx(1.23)
    assert utils.convert_ras_hdf_value(np.int32(123)) == 123
    assert utils.convert_ras_hdf_value(b"15Mar2024 16:39:01") == datetime(
        2024, 3, 15, 16, 39, 1
    )
    assert utils.convert_ras_hdf_value(b"15Mar2024 24:00:00") == datetime(
        2024, 3, 16, 0, 0, 0
    )
    assert utils.convert_ras_hdf_value(b"15Mar2024 16:39:01 to 16Mar2024 16:39:01") == [
        datetime(2024, 3, 15, 16, 39, 1),
        datetime(2024, 3, 16, 16, 39, 1),
    ]
    assert utils.convert_ras_hdf_value(b"18Mar2024 24:00:00 to 19Mar2024 24:00:00") == [
        datetime(2024, 3, 19, 0, 0, 0),
        datetime(2024, 3, 20, 0, 0, 0),
    ]
    assert utils.convert_ras_hdf_value(b"01:23:45") == timedelta(
        hours=1, minutes=23, seconds=45
    )
    assert utils.convert_ras_hdf_value(b"15Mar2024 2400") == datetime(
        2024, 3, 16, 0, 0, 0
    )
    assert utils.convert_ras_hdf_value(b"15Mar2024 2315") == datetime(
        2024, 3, 15, 23, 15, 0
    )

    assert utils.convert_ras_hdf_value(b"15Mar2024 1639 to 16Mar2024 1639") == [
        datetime(2024, 3, 15, 16, 39, 0),
        datetime(2024, 3, 16, 16, 39, 0),
    ]
    assert utils.convert_ras_hdf_value(b"18Mar2024 2400 to 19Mar2024 2400") == [
        datetime(2024, 3, 19, 0, 0, 0),
        datetime(2024, 3, 20, 0, 0, 0),
    ]
