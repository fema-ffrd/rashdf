from src.rashdf import utils

import numpy as np
import pytest

from datetime import datetime


def test_convert_ras_hdf_value():
    assert utils.convert_ras_hdf_value(b"True") == True
    assert utils.convert_ras_hdf_value(b"False") == False
    assert utils.convert_ras_hdf_value(np.float32(1.23)) == pytest.approx(1.23)
    assert utils.convert_ras_hdf_value(np.int32(123)) == 123
    assert utils.convert_ras_hdf_value(b"15Mar2024 16:39:01") == datetime(2024, 3, 15, 16, 39, 1)
    assert utils.convert_ras_hdf_value(b"15Mar2024 16:39:01 to 16Mar2024 16:39:01") == [
        datetime(2024, 3, 15, 16, 39, 1),
        datetime(2024, 3, 16, 16, 39, 1),
    ]
