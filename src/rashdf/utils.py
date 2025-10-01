"""Utility functions for reading HEC-RAS HDF data."""

import h5py
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
import re
from typing import Any, Callable, List, Tuple, Union, Optional
import warnings
from shapely import LineString, MultiLineString
import geopandas as gpd


def experimental(func) -> Callable:
    """
    Declare a function to be experimental.

    This is a decorator which can be used to mark functions as experimental.
    It will result in a warning being emitted when the function is used.

    Parameters
    ----------
        func: The function to be declared experimental.

    Returns
    -------
        The decorated function.
    """

    def new_func(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is experimental and could change in the future. Please review output carefully.",
            category=UserWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


def deprecated(func) -> Callable:
    """
    Deprecate a function.

    This is a decorator which can be used to mark functions as deprecated.
    It will result in a warning being emitted when the function is used.

    Parameters
    ----------
        func: The function to be deprecated.

    Returns
    -------
        The decorated function.
    """

    def new_func(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is deprecated and will be removed in a future version.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


def parse_ras_datetime_ms(datetime_str: str) -> datetime:
    """Parse a datetime string with milliseconds from a RAS file into a datetime object.

    If the datetime has a time of 2400, then it is converted to midnight of the next day.

    Parameters
    ----------
        datetime_str (str): The datetime string to be parsed. The string should be in the format "ddMMMyyyy HH:mm:ss:fff".

    Returns
    -------
        datetime: A datetime object representing the parsed datetime.
    """
    milliseconds = int(datetime_str[-3:])
    microseconds = milliseconds * 1000
    parsed_dt = parse_ras_datetime(datetime_str[:-4]).replace(microsecond=microseconds)
    return parsed_dt


def parse_ras_datetime(datetime_str: str) -> datetime:
    """Parse a datetime string from a RAS file into a datetime object.

    If the datetime has a time of 2400, then it is converted to midnight of the next day.

    Parameters
    ----------
        datetime_str (str): The datetime string to be parsed.

    Returns
    -------
        datetime: A datetime object representing the parsed datetime.
    """
    date_formats = ["%d%b%Y", "%m/%d/%Y", "%m-%d-%Y", "%Y/%m/%d", "%Y-%m-%d"]
    time_formats = ["%H:%M:%S", "%H%M"]
    datetime_formats = [
        f"{date} {time}" for date in date_formats for time in time_formats
    ]

    is_2400 = datetime_str.endswith((" 24:00:00", " 2400", " 24:00"))
    if is_2400:
        datetime_str = datetime_str.split()[0] + " 00:00:00"

    last_exception = None
    for fmt in datetime_formats:
        try:
            parsed_dt = datetime.strptime(datetime_str, fmt)
            if is_2400:
                parsed_dt += timedelta(days=1)
            return parsed_dt
        except ValueError as e:
            last_exception = e
            continue

    raise ValueError(f"Invalid date format: {datetime_str}") from last_exception


@deprecated
def parse_ras_simulation_window_datetime(datetime_str) -> datetime:
    """
    Parse a datetime string from a RAS simulation window into a datetime object.

    If the datetime has a time of 2400, then it is converted to midnight of the next day.

    Parameters
    ----------
        datetime_str: The datetime string to be parsed.

    Returns
    -------
        datetime: A datetime object representing the parsed datetime.
    """
    datetime_format = "%d%b%Y %H%M"

    if datetime_str.endswith("2400"):
        datetime_str = datetime_str.replace("2400", "0000")
        parsed_dt = datetime.strptime(datetime_str, datetime_format)
        parsed_dt += timedelta(days=1)
    else:
        parsed_dt = datetime.strptime(datetime_str, datetime_format)

    return parsed_dt


def parse_run_time_window(window: str) -> Tuple[datetime, datetime]:
    """
    Parse a run time window string into a tuple of datetime objects.

    Parameters
    ----------
        window (str): The run time window string to be parsed.

    Returns
    -------
        Tuple[datetime, datetime]: A tuple containing two datetime objects representing the start and end of the run
        time window.
    """
    split = window.split(" to ")
    begin = parse_ras_datetime(split[0])
    end = parse_ras_datetime(split[1])
    return begin, end


def parse_duration(duration_str: str) -> timedelta:
    """
    Parse a duration string into a timedelta object.

    Parameters
    ----------
        duration_str (str): The duration string to be parsed. The string should be in the format "HH:MM:SS".

    Returns
    -------
        timedelta: A timedelta object representing the parsed duration.
    """
    hours, minutes, seconds = map(int, duration_str.split(":"))
    duration = timedelta(hours=hours, minutes=minutes, seconds=seconds)
    return duration


def convert_ras_hdf_string(
    value: str,
) -> Union[bool, datetime, List[datetime], timedelta, str]:
    """Convert a string value from an HEC-RAS HDF file into a Python object.

    This function handles several specific string formats:
    - The strings "True" and "False" are converted to boolean values.
    - Strings matching the format "ddMMMyyyy HH:mm:ss" or "ddMMMyyyy HHmm"
      are parsed into datetime objects.
    - Strings matching the format "ddMMMyyyy HH:mm:ss to ddMMMyyyy HH:mm:ss" or
      "ddMMMyyyy HHmm to ddMMMyyyy HHmm" are parsed into a list of two datetime objects.
    - Strings matching the format "HH:mm:ss" are parsed into timedelta objects.

    Parameters
    ----------
        value (str): The string value to be converted.

    Returns
    -------
        The converted value, which could be a boolean, a datetime string,
        a list of datetime strings, a timedelta objects, or the original string
        if no other conditions are met.
    """
    ras_duration_format_re = r"\d{2}:\d{2}:\d{2}"
    date_patterns_re = [
        r"\d{2}\w{3}\d{4}",
        r"\d{2}/\d{2}/\d{4}",
        r"\d{2}-\d{2}-\d{4}",
        r"\d{4}/\d{2}/\d{2}",
        r"\d{4}-\d{2}-\d{2}",
    ]
    time_patterns_re = [
        r"\d{2}:\d{2}:\d{2}",
        r"\d{4}",
    ]
    datetime_patterns_re = [
        f"{date} {time}" for date in date_patterns_re for time in time_patterns_re
    ]
    s = value.decode("utf-8")
    if s == "True":
        return True
    elif s == "False":
        return False
    elif re.match(rf"^{ras_duration_format_re}$", s):
        return parse_duration(s)
    for dt_re in datetime_patterns_re:
        if re.match(rf"^{dt_re}", s):
            if re.match(rf"^{dt_re} to {dt_re}$", s):
                start, end = s.split(" to ")
                return [
                    parse_ras_datetime(start),
                    parse_ras_datetime(end),
                ]
            return parse_ras_datetime(s)
    return s


def convert_ras_hdf_value(
    value: Any,
) -> Union[None, bool, str, List[str], int, float, List[int], List[float]]:
    """Convert a value from a HEC-RAS HDF file into a Python object.

    This function handles several specific types:
    - NaN values are converted to None.
    - Byte strings are converted using the `convert_ras_hdf_string` function.
    - NumPy integer or float types are converted to Python int or float.
    - Regular ints and floats are left as they are.
    - Lists, tuples, and NumPy arrays are recursively processed.
    - All other types are converted to string.

    Parameters
    ----------
        value (Any): The value to be converted.

    Returns
    -------
        The converted value, which could be None, a boolean, a string, a list of strings, an integer, a float, a list
        of integers, a list of floats, or the original value as a string if no other conditions are met.
    """
    # Check for NaN (np.nan)
    if isinstance(value, np.floating) and np.isnan(value):
        return None

    # Check for byte strings
    elif isinstance(value, bytes) or isinstance(value, np.bytes_):
        return convert_ras_hdf_string(value)

    # Check for NumPy integer or float types
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)

    # Leave regular ints and floats as they are
    elif isinstance(value, (int, float)):
        return value

    elif isinstance(value, (list, tuple, np.ndarray)):
        if len(value) > 1:
            return [convert_ras_hdf_value(v) for v in value]
        else:
            return convert_ras_hdf_value(value[0])

    # Convert all other types to string
    else:
        return str(value)


def hdf5_attrs_to_dict(attrs: dict, prefix: str = None) -> dict:
    """
    Convert a dictionary of attributes from an HDF5 file into a Python dictionary.

    Parameters
    ----------
        attrs (dict): The attributes to be converted.
        prefix (str, optional): An optional prefix to prepend to the keys.

    Returns
    -------
        dict: A dictionary with the converted attributes.
    """
    results = {}
    for k, v in attrs.items():
        value = convert_ras_hdf_value(v)
        if prefix:
            key = f"{prefix}:{k}"
        else:
            key = k
        results[key] = value
    return results


def get_first_hdf_group(parent_group: h5py.Group) -> Optional[h5py.Group]:
    """
    Get the first HDF5 group from a parent group.

    This function iterates over the items in the parent group and returns the first item that is an instance of
     h5py.Group. If no such item is found, it returns None.

    Parameters
    ----------
        parent_group (h5py.Group): The parent group to search in.

    Returns
    -------
        Optional[h5py.Group]: The first HDF5 group in the parent group, or None if no group is found.
    """
    for _, item in parent_group.items():
        if isinstance(item, h5py.Group):
            return item
    return None


def df_datetimes_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any datetime64 columns in a DataFrame to strings.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to convert.

    Returns
    -------
    DataFrame
        The DataFrame with any datetime64 columns converted to strings.
    """
    df_result = df.copy()
    for col in df.select_dtypes(include=["datetime64"]).columns:
        df_result[col] = df[col].apply(
            lambda x: pd.Timestamp(x).isoformat() if pd.notnull(x) else None
        )
    return df_result


def ras_timesteps_to_datetimes(
    timesteps: np.ndarray, start_time: datetime, time_unit: str, round_to="0.1 s"
) -> List[datetime]:
    """
    Convert an array of RAS timesteps into an array of datetime objects.

    Parameters
    ----------
        timesteps (np.ndarray): An array of RAS timesteps.
        start_time (datetime): The start time of the simulation.
        time_unit (str): The time unit of the timesteps.
        round_to (str): The time unit to round the datetimes to. (Default: "0.1 s")

    Returns
    -------
        List[datetime]: A list of datetime objects corresponding to the timesteps.
    """
    return [
        start_time + pd.Timedelta(timestep, unit=time_unit).round(round_to)
        for timestep in timesteps.astype(np.float64)
    ]


def remove_line_ends(
    geom: Union[LineString, MultiLineString],
) -> Union[LineString, MultiLineString]:
    """
    Remove endpoints from a LineString or each LineString in a MultiLineString if longer than 3 points.

    Parameters
    ----------
    geom : LineString or MultiLineString
        The geometry to trim.

    Returns
    -------
    LineString or MultiLineString
        The trimmed geometry, or original if not enough points to trim.
    """
    if isinstance(geom, LineString):
        coords = list(geom.coords)
        if len(coords) > 3:
            return LineString(coords[1:-1])
        return geom
    elif isinstance(geom, MultiLineString):
        trimmed = []
        for line in geom.geoms:
            coords = list(line.coords)
            if len(coords) > 3:
                trimmed.append(LineString(coords[1:-1]))
            else:
                trimmed.append(line)
        return MultiLineString(trimmed)
    return geom


def reverse_line(
    line: Union[LineString, MultiLineString],
) -> Union[LineString, MultiLineString]:
    """
    Reverse the order of coordinates in a LineString or each LineString in a MultiLineString.

    Parameters
    ----------
    line : LineString or MultiLineString
        The geometry to reverse.

    Returns
    -------
    LineString or MultiLineString
        The reversed geometry.
    """
    return (
        MultiLineString([LineString(list(line.coords)[::-1]) for line in line.geoms])
        if isinstance(line, MultiLineString)
        else LineString(list(line.coords)[::-1])
    )


def copy_lines_parallel(
    lines: gpd.GeoDataFrame,
    offset_ft: Union[np.ndarray, float],
    id_col: str = "id",
) -> gpd.GeoDataFrame:
    """
    Create parallel copies of line geometries offset to the left and right, then trim and erase overlaps.

    Parameters
    ----------
    lines : gpd.GeoDataFrame
        GeoDataFrame containing line geometries.
    offset_ft : float or np.ndarray
        Offset distance (in feet) for parallel lines.
    id_col : str
        Name of the column containing unique structure IDs. Default is "id".

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with trimmed, parallel left and right offset lines.
    """
    # Offset lines to the left
    left = lines.copy()
    offset_ft = offset_ft.astype(float)
    left.geometry = lines.buffer(
        offset_ft, cap_style="flat", single_sided=True, resolution=3
    ).boundary
    left["side"] = "left"

    # Offset lines to the right (reverse direction first)
    reversed_lines = lines.copy()
    reversed_lines.geometry = reversed_lines.geometry.apply(reverse_line)
    right = lines.copy()
    right.geometry = reversed_lines.buffer(
        offset_ft, cap_style="flat", single_sided=True, resolution=3
    ).boundary.apply(reverse_line)
    right["side"] = "right"

    # Combine left and right boundaries
    boundaries = pd.concat([left, right], ignore_index=True)
    boundaries_gdf = gpd.GeoDataFrame(boundaries, crs=lines.crs, geometry="geometry")

    # Erase buffer caps
    erase_buffer = 0.1
    cleaned_list = []
    eraser = gpd.GeoDataFrame(
        {
            id_col: lines[id_col],
            "geometry": lines.buffer(
                offset_ft - erase_buffer, cap_style="square", resolution=3
            ),
        },
        crs=lines.crs,
    )
    for id in lines[id_col].unique():
        cleaned_list.append(
            gpd.overlay(
                boundaries_gdf[boundaries_gdf[id_col] == id],
                eraser[eraser[id_col] == id],
                how="difference",
            )
        )
    cleaned = gpd.GeoDataFrame(
        pd.concat(cleaned_list, ignore_index=True), crs=lines.crs, geometry="geometry"
    )

    # trim ends
    cleaned["geometry"] = cleaned["geometry"].apply(remove_line_ends)
    return cleaned
