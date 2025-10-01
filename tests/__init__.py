import h5py
from geopandas import GeoDataFrame
import json
import hashlib
from pathlib import Path
import numpy as np


def _create_hdf_with_group_attrs(path: Path, group_path: str, attrs: dict):
    with h5py.File(path, "w") as f:
        group = f.create_group(group_path)
        for key, value in attrs.items():
            group.attrs[key] = value


def _gdf_matches_json(gdf: GeoDataFrame, json_file: Path) -> bool:
    with open(json_file) as j:
        return gdf.to_json() == j.read()


def get_sha1_hash(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def _gdf_matches_json_alt(gdf: GeoDataFrame, json_file: Path) -> bool:
    with open(json_file) as j:
        return json.loads(gdf.to_json()) == json.load(j)


def _assert_geodataframes_close(
    gdf1: GeoDataFrame, gdf2: GeoDataFrame, tol: float = 1e-3
) -> None:
    assert gdf1.crs == gdf2.crs
    assert gdf1.shape == gdf2.shape
    assert set(gdf1.columns) == set(gdf2.columns)
    for col in gdf1.columns:
        if col == gdf1.geometry.name:
            geom_close = []
            for geom1, geom2 in zip(gdf1.geometry, gdf2.geometry):
                if not geom1.equals_exact(geom2, tolerance=tol):
                    geom_close.append(False)
                else:
                    geom_close.append(True)
            assert all(geom_close)
        else:
            try:
                all_close = np.allclose(
                    gdf1[col].values, gdf2[col].values, atol=tol, equal_nan=True
                )
                assert all_close
            except TypeError:
                assert (gdf1[col].values == gdf2[col].values).all()
