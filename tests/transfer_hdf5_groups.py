#!/usr/bin/env python3
"""
Transfer datasets and attributes from a nested HDF5 group to another HDF5 file.
Includes parent groups and their attributes.
"""

import h5py
import argparse
from pathlib import Path


def copy_attributes(source_obj, dest_obj):
    """Copy all attributes from source object to destination object."""
    for attr_name, attr_value in source_obj.attrs.items():
        dest_obj.attrs[attr_name] = attr_value


def copy_group_recursive(source_group, dest_group, ignore_groups=None):
    """Recursively copy all datasets and subgroups from source to destination."""
    if ignore_groups is None:
        ignore_groups = set()

    # Copy attributes of current group
    copy_attributes(source_group, dest_group)

    # Copy all items in the group
    for key, item in source_group.items():
        # Skip ignored groups
        if key in ignore_groups:
            print(f"Skipping ignored group: {key}")
            continue

        print(key, item)
        if isinstance(item, h5py.Dataset):
            # Copy dataset with data and attributes
            dest_group.create_dataset(key, data=item[0:24])
            copy_attributes(item, dest_group[key])
        elif isinstance(item, h5py.Group):
            # Create subgroup and recursively copy
            sub_group = dest_group.create_group(key)
            copy_group_recursive(item, sub_group, ignore_groups)


def get_group_path_parts(group_path):
    """Convert a group path string to a list of group names."""
    path = group_path.strip("/")
    return path.split("/") if path else []


def ensure_parent_groups(hdf5_file, group_path):
    """
    Ensure all parent groups exist in the HDF5 file.
    Returns the leaf group, creating parent groups if needed.
    """
    parts = get_group_path_parts(group_path)
    current = hdf5_file

    for part in parts:
        if part not in current:
            current.create_group(part)
        current = current[part]

    return current


def transfer_group(
    source_file_path,
    source_group_path,
    dest_file_path,
    dest_group_path=None,
    mode="a",
    ignore_groups=None,
):
    """
    Transfer a nested group from source HDF5 file to destination HDF5 file.

    Args:
        source_file_path (str): Path to source HDF5 file
        source_group_path (str): Path to group within source file (e.g., '/path/to/group')
        dest_file_path (str): Path to destination HDF5 file
        dest_group_path (str, optional): Path where to place group in destination.
                                        If None, uses source_group_path
        mode (str): HDF5 file open mode for destination ('r+' for existing, 'a' for create if needed)
        ignore_groups (set, optional): Set of group names to ignore during transfer
    """
    dest_group_path = dest_group_path or source_group_path
    if ignore_groups is None:
        ignore_groups = set()

    with h5py.File(source_file_path, "r") as src_file:
        if source_group_path not in src_file:
            raise ValueError(f"Group '{source_group_path}' not found in source file")

        source_group = src_file[source_group_path]

        with h5py.File(dest_file_path, mode) as dest_file:
            # Ensure all parent groups exist
            dest_group = ensure_parent_groups(dest_file, dest_group_path)

            # Copy the source group's content
            copy_group_recursive(source_group, dest_group, ignore_groups)

    print(
        f"✓ Successfully transferred '{source_group_path}' to '{dest_group_path}' in {dest_file_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Transfer HDF5 group with parent hierarchy and attributes"
    )
    parser.add_argument("source_file", help="Source HDF5 file path")
    parser.add_argument("source_group", help="Source group path (e.g., /path/to/group)")
    parser.add_argument("dest_file", help="Destination HDF5 file path")
    parser.add_argument(
        "--dest-group",
        default=None,
        help="Destination group path (defaults to source group path)",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create destination file if it does not exist",
    )
    parser.add_argument(
        "--ignore",
        nargs="+",
        default=[],
        help="Group names to ignore during transfer (space-separated)",
    )

    args = parser.parse_args()

    # Determine file mode
    if args.create or not Path(args.dest_file).exists():
        mode = "w" if not Path(args.dest_file).exists() else "a"
    else:
        mode = "r+"

    # Convert ignore list to set for efficient lookup
    ignore_groups = set(args.ignore)

    try:
        transfer_group(
            args.source_file,
            args.source_group,
            args.dest_file,
            args.dest_group,
            mode=mode,
            ignore_groups=ignore_groups,
        )
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
