from src.rashdf.base import RasHdf
from unittest.mock import patch


def test_open():
    rasfile = "Muncie.g05.hdf"
    rasfile_path = f"./tests/data/ras/{rasfile}"
    hdf = RasHdf(rasfile_path)
    assert hdf._loc == rasfile_path


def test_open_uri():
    rasfile = "Muncie.g05.hdf"
    rasfile_path = f"./tests/data/ras/{rasfile}"
    url = f"s3://mybucket/{rasfile}"

    # Mock the specific functions used by s3fs
    with patch("s3fs.core.S3FileSystem.open", return_value=open(rasfile_path, "rb")):
        hdf = RasHdf.open_uri(url)
        assert hdf._loc == url
