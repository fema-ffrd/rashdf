import h5py

from pathlib import Path


def _create_hdf_with_group_attrs(path: Path, group_path: str, attrs: dict):
    with h5py.File(path, "w") as f:
        group = f.create_group(group_path)
        for key, value in attrs.items():
            group.attrs[key] = value
