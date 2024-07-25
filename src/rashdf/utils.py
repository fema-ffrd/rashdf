"""Utility functions for reading HEC-RAS HDF data."""

import h5py
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
import re
from typing import Any, Callable, List, Tuple, Union, Optional
import warnings


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
        datetime_str (str): The datetime string to be parsed. The string should be in the format "ddMMMyyyy HH:mm:ss".

    Returns
    -------
        datetime: A datetime object representing the parsed datetime.
    """
    format = "%d%b%Y %H:%M:%S"

    if datetime_str.endswith("24:00:00"):
        datetime_str = datetime_str.replace("24:00:00", "00:00:00")
        parsed_dt = datetime.strptime(datetime_str, format)
        parsed_dt += timedelta(days=1)
    else:
        parsed_dt = datetime.strptime(datetime_str, format)

    return parsed_dt


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
    format = "%d%b%Y %H%M"

    if datetime_str.endswith("2400"):
        datetime_str = datetime_str.replace("2400", "0000")
        parsed_dt = datetime.strptime(datetime_str, format)
        parsed_dt += timedelta(days=1)
    else:
        parsed_dt = datetime.strptime(datetime_str, format)

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
    ras_datetime_format1_re = r"\d{2}\w{3}\d{4} \d{2}:\d{2}:\d{2}"
    ras_datetime_format2_re = r"\d{2}\w{3}\d{4} \d{2}\d{2}"
    ras_duration_format_re = r"\d{2}:\d{2}:\d{2}"
    s = value.decode("utf-8")
    if s == "True":
        return True
    elif s == "False":
        return False
    elif re.match(rf"^{ras_datetime_format1_re}", s):
        if re.match(rf"^{ras_datetime_format1_re} to {ras_datetime_format1_re}$", s):
            split = s.split(" to ")
            return [
                parse_ras_datetime(split[0]),
                parse_ras_datetime(split[1]),
            ]
        return parse_ras_datetime(s)
    elif re.match(rf"^{ras_datetime_format2_re}", s):
        if re.match(rf"^{ras_datetime_format2_re} to {ras_datetime_format2_re}$", s):
            split = s.split(" to ")
            return [
                parse_ras_simulation_window_datetime(split[0]),
                parse_ras_simulation_window_datetime(split[1]),
            ]
        return parse_ras_simulation_window_datetime(s)
    elif re.match(rf"^{ras_duration_format_re}$", s):
        return parse_duration(s)
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
    # TODO (?): handle "8-bit bitfield" values in 2D Flow Area groups

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
