from src.rashdf import utils

import numpy as np
import pandas as pd
import pytest

from datetime import datetime, timedelta
from shapely.geometry import LineString, MultiLineString
import geopandas as gpd
from pathlib import Path

from . import _assert_geodataframes_close

TEST_DATA = Path("./tests/data")
TEST_JSON = TEST_DATA / "json"


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
    assert utils.convert_ras_hdf_value(b"15Mar2024 2400") == datetime(
        2024, 3, 16, 0, 0, 0
    )
    assert utils.convert_ras_hdf_value(b"03/15/2024 2400") == datetime(
        2024, 3, 16, 0, 0, 0
    )
    assert utils.convert_ras_hdf_value(b"03-15-2024 2400") == datetime(
        2024, 3, 16, 0, 0, 0
    )
    assert utils.convert_ras_hdf_value(b"2024/03/15 2400") == datetime(
        2024, 3, 16, 0, 0, 0
    )
    assert utils.convert_ras_hdf_value(b"2024-03-15 2400") == datetime(
        2024, 3, 16, 0, 0, 0
    )
    assert utils.convert_ras_hdf_value(b"15Mar2024 0000") == datetime(
        2024, 3, 15, 0, 0, 0
    )
    assert utils.convert_ras_hdf_value(b"03/15/2024 0000") == datetime(
        2024, 3, 15, 0, 0, 0
    )
    assert utils.convert_ras_hdf_value(b"03-15-2024 0000") == datetime(
        2024, 3, 15, 0, 0, 0
    )
    assert utils.convert_ras_hdf_value(b"2024/03/15 0000") == datetime(
        2024, 3, 15, 0, 0, 0
    )
    assert utils.convert_ras_hdf_value(b"2024-03-15 0000") == datetime(
        2024, 3, 15, 0, 0, 0
    )
    assert utils.convert_ras_hdf_value(b"15Mar2024 23:59:59") == datetime(
        2024, 3, 15, 23, 59, 59
    )
    assert utils.convert_ras_hdf_value(b"03/15/2024 23:59:59") == datetime(
        2024, 3, 15, 23, 59, 59
    )
    assert utils.convert_ras_hdf_value(b"03-15-2024 23:59:59") == datetime(
        2024, 3, 15, 23, 59, 59
    )
    assert utils.convert_ras_hdf_value(b"2024/03/15 23:59:59") == datetime(
        2024, 3, 15, 23, 59, 59
    )
    assert utils.convert_ras_hdf_value(b"2024-03-15 23:59:59") == datetime(
        2024, 3, 15, 23, 59, 59
    )
    assert utils.convert_ras_hdf_value(b"15Mar2024 1639 to 16Mar2024 1639") == [
        datetime(2024, 3, 15, 16, 39, 0),
        datetime(2024, 3, 16, 16, 39, 0),
    ]
    assert utils.convert_ras_hdf_value(b"18Mar2024 2400 to 19Mar2024 2400") == [
        datetime(2024, 3, 19, 0, 0, 0),
        datetime(2024, 3, 20, 0, 0, 0),
    ]


def test_df_datetimes_to_str():
    df = pd.DataFrame(
        {
            "datetime": [
                datetime(2024, 3, 15, 16, 39, 1),
                datetime(2024, 3, 16, 16, 39, 1),
            ],
            "asdf": [
                0.123,
                0.456,
            ],
        }
    )
    assert df["datetime"].dtype.name == "datetime64[ns]"
    df = utils.df_datetimes_to_str(df)
    assert df["datetime"].dtype.name == "object"
    assert df["datetime"].tolist() == ["2024-03-15T16:39:01", "2024-03-16T16:39:01"]
    assert df["asdf"].tolist() == [0.123, 0.456]


def test_parse_ras_datetime_ms():
    assert utils.parse_ras_datetime_ms("15Mar2024 16:39:01.123") == datetime(
        2024, 3, 15, 16, 39, 1, 123000
    )
    assert utils.parse_ras_datetime_ms("15Mar2024 24:00:00.000") == datetime(
        2024, 3, 16, 0, 0, 0, 0
    )


def test_trim_line():
    gdf = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3],
            "geometry": [
                LineString([(0, 0), (5, 5), (10, 10)]),
                LineString([(0, 0), (5, 5), (10, 10), (15, 15)]),
                MultiLineString(
                    [
                        [(0, 0), (3, 3)],
                        [(3, 3), (6, 6), (9, 9)],
                        [(3, 3), (6, 6), (9, 9), (12, 12), (15, 15)],
                    ]
                ),
            ],
        },
    )
    assert gdf.geometry.apply(utils.remove_line_ends).equals(
        gpd.GeoSeries(
            [
                LineString([(0, 0), (5, 5), (10, 10)]),
                LineString(
                    [
                        (5, 5),
                        (10, 10),
                    ]
                ),
                MultiLineString(
                    [
                        [(0, 0), (3, 3)],
                        [(3, 3), (6, 6), (9, 9)],
                        [(6, 6), (9, 9), (12, 12)],
                    ]
                ),
            ]
        )
    )


def test_reverse_line():
    gdf = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3],
            "geometry": [
                LineString([(0, 0), (5, 5), (10, 10)]),
                LineString([(0, 0), (5, 5), (10, 10), (15, 15)]),
                MultiLineString(
                    [
                        [(0, 0), (3, 3)],
                        [(3, 3), (6, 6), (9, 9)],
                        [(3, 3), (6, 6), (9, 9), (12, 12), (15, 15)],
                    ]
                ),
            ],
        },
    )
    assert gdf.geometry.apply(utils.reverse_line).equals(
        gpd.GeoSeries(
            [
                LineString([(10, 10), (5, 5), (0, 0)]),
                LineString([(15, 15), (10, 10), (5, 5), (0, 0)]),
                MultiLineString(
                    [
                        [(3, 3), (0, 0)],
                        [(9, 9), (6, 6), (3, 3)],
                        [(15, 15), (12, 12), (9, 9), (6, 6), (3, 3)],
                    ]
                ),
            ]
        )
    )


def test_copy_lines_parallel():
    gdf = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3],
            "geometry": [
                LineString([(0, 0), (5, 5), (10, 10)]),
                LineString([(20, 20), (30, 30), (40, 40), (50, 50)]),
                MultiLineString(
                    [
                        [(100.0, 100.0), (103.0, 103.0)],
                        [(103.0, 103.0), (106.0, 106.0), (109.0, 109.0)],
                        [
                            (103.0, 103.0),
                            (106.0, 106.0),
                            (109.0, 109.0),
                            (112.0, 112.0),
                            (115.0, 115.0),
                        ],
                    ]
                ),
            ],
        },
    )
    offsets = np.array([1, 2, 3])
    copied = utils.copy_lines_parallel(gdf, offsets)
    expected = gpd.read_file(TEST_JSON / "copy_lines_parallel.json").set_crs(
        None, allow_override=True
    )
    _assert_geodataframes_close(copied, expected, tol=1e-3)
