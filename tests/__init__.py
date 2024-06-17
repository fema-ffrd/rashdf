import h5py
from geopandas import GeoDataFrame

import hashlib
from pathlib import Path


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
