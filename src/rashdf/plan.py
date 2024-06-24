"""HEC-RAS Plan HDF class."""

from .geom import RasGeomHdf
from .utils import (
    df_datetimes_to_str,
    ras_timesteps_to_datetimes,
    parse_ras_datetime_ms,
)

from geopandas import GeoDataFrame
import h5py
import numpy as np
from pandas import DataFrame
import pandas as pd
import xarray as xr

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


class RasPlanHdfError(Exception):
    """HEC-RAS Plan HDF error class."""

    pass


class XsSteadyOutputVar(Enum):
    """Summary of steady cross section output variables."""

    ENERGY_GRADE = "Energy Grade"
    FLOW = "Flow"
    WATER_SURFACE = "Water Surface"
    ENCROACHMENT_STATION_LEFT = "Encroachment Station Left"
    ENCROACHMENT_STATION_RIGHT = "Encroachment Station Right"
    AREA_INEFFECTIVE_TOTAL = "Area including Ineffective Total"
    VELOCITY_TOTAL = "Velocity Total"


XS_STEADY_OUTPUT_ADDITIONAL = [
    XsSteadyOutputVar.ENCROACHMENT_STATION_LEFT,
    XsSteadyOutputVar.ENCROACHMENT_STATION_RIGHT,
    XsSteadyOutputVar.AREA_INEFFECTIVE_TOTAL,
    XsSteadyOutputVar.VELOCITY_TOTAL,
]


class SummaryOutputVar(Enum):
    """Summary output variables."""

    MAXIMUM_WATER_SURFACE = "Maximum Water Surface"
    MINIMUM_WATER_SURFACE = "Minimum Water Surface"
    MAXIMUM_FACE_VELOCITY = "Maximum Face Velocity"
    MINIMUM_FACE_VELOCITY = "Minimum Face Velocity"
    CELL_MAXIMUM_WATER_SURFACE_ERROR = "Cell Maximum Water Surface Error"
    CELL_CUMULATIVE_ITERATION = "Cell Cumulative Iteration"
    CELL_LAST_ITERATION = "Cell Last Iteration"


SUMMARY_OUTPUT_VARS_CELLS = [
    SummaryOutputVar.MAXIMUM_WATER_SURFACE,
    SummaryOutputVar.MINIMUM_WATER_SURFACE,
    SummaryOutputVar.CELL_MAXIMUM_WATER_SURFACE_ERROR,
    SummaryOutputVar.CELL_CUMULATIVE_ITERATION,
    SummaryOutputVar.CELL_LAST_ITERATION,
]

SUMMARY_OUTPUT_VARS_FACES = [
    SummaryOutputVar.MAXIMUM_FACE_VELOCITY,
    SummaryOutputVar.MINIMUM_FACE_VELOCITY,
]


class TimeSeriesOutputVar(Enum):
    """Time series output variables."""

    # Default Outputs
    WATER_SURFACE = "Water Surface"
    FACE_VELOCITY = "Face Velocity"

    # Optional Outputs
    CELL_COURANT = "Cell Courant"
    CELL_CUMULATIVE_PRECIPITATION_DEPTH = "Cell Cumulative Precipitation Depth"
    CELL_DIVERGENCE_TERM = "Cell Divergence Term"
    CELL_EDDY_VISCOSITY_X = "Cell Eddy Viscosity - Eddy Viscosity X"
    CELL_EDDY_VISCOSITY_Y = "Cell Eddy Viscosity - Eddy Viscosity Y"
    CELL_FLOW_BALANCE = "Cell Flow Balance"
    CELL_HYDRAULIC_DEPTH = "Cell Hydraulic Depth"
    CELL_INVERT_DEPTH = "Cell Invert Depth"
    CELL_STORAGE_TERM = "Cell Storage Term"
    CELL_VELOCITY_X = "Cell Velocity - Velocity X"
    CELL_VELOCITY_Y = "Cell Velocity - Velocity Y"
    CELL_VOLUME = "Cell Volume"
    CELL_VOLUME_ERROR = "Cell Volume Error"
    CELL_WATER_SOURCE_TERM = "Cell Water Source Term"
    CELL_WATER_SURFACE_ERROR = "Cell Water Surface Error"

    FACE_COURANT = "Face Courant"
    FACE_CUMULATIVE_VOLUME = "Face Cumulative Volume"
    FACE_EDDY_VISCOSITY = "Face Eddy Viscosity"
    FACE_FLOW = "Face Flow"
    FACE_FLOW_PERIOD_AVERAGE = "Face Flow Period Average"
    FACE_FRICTION_TERM = "Face Friction Term"
    FACE_PRESSURE_GRADIENT_TERM = "Face Pressure Gradient Term"
    FACE_SHEAR_STRESS = "Face Shear Stress"
    FACE_TANGENTIAL_VELOCITY = "Face Tangential Velocity"
    FACE_WATER_SURFACE = "Face Water Surface"
    FACE_WIND_TERM = "Face Wind Term"


TIME_SERIES_OUTPUT_VARS_CELLS = [
    TimeSeriesOutputVar.WATER_SURFACE,
    TimeSeriesOutputVar.CELL_COURANT,
    TimeSeriesOutputVar.CELL_CUMULATIVE_PRECIPITATION_DEPTH,
    TimeSeriesOutputVar.CELL_DIVERGENCE_TERM,
    TimeSeriesOutputVar.CELL_EDDY_VISCOSITY_X,
    TimeSeriesOutputVar.CELL_EDDY_VISCOSITY_Y,
    TimeSeriesOutputVar.CELL_FLOW_BALANCE,
    TimeSeriesOutputVar.CELL_HYDRAULIC_DEPTH,
    TimeSeriesOutputVar.CELL_INVERT_DEPTH,
    TimeSeriesOutputVar.CELL_STORAGE_TERM,
    TimeSeriesOutputVar.CELL_VELOCITY_X,
    TimeSeriesOutputVar.CELL_VELOCITY_Y,
    TimeSeriesOutputVar.CELL_VOLUME,
    TimeSeriesOutputVar.CELL_VOLUME_ERROR,
    TimeSeriesOutputVar.CELL_WATER_SOURCE_TERM,
    TimeSeriesOutputVar.CELL_WATER_SURFACE_ERROR,
]

TIME_SERIES_OUTPUT_VARS_FACES = [
    TimeSeriesOutputVar.FACE_COURANT,
    TimeSeriesOutputVar.FACE_CUMULATIVE_VOLUME,
    TimeSeriesOutputVar.FACE_EDDY_VISCOSITY,
    TimeSeriesOutputVar.FACE_FLOW,
    TimeSeriesOutputVar.FACE_FLOW_PERIOD_AVERAGE,
    TimeSeriesOutputVar.FACE_FRICTION_TERM,
    TimeSeriesOutputVar.FACE_PRESSURE_GRADIENT_TERM,
    TimeSeriesOutputVar.FACE_SHEAR_STRESS,
    TimeSeriesOutputVar.FACE_TANGENTIAL_VELOCITY,
    TimeSeriesOutputVar.FACE_VELOCITY,
    TimeSeriesOutputVar.FACE_WATER_SURFACE,
    # TODO: investigate why "Face Wind Term" data gets written as a 1D array
    # TimeSeriesOutputVar.FACE_WIND_TERM,
]

TIME_SERIES_OUTPUT_VARS_DEFAULT = [
    TimeSeriesOutputVar.WATER_SURFACE,
    TimeSeriesOutputVar.FACE_VELOCITY,
]


class RasPlanHdf(RasGeomHdf):
    """HEC-RAS Plan HDF class."""

    PLAN_INFO_PATH = "Plan Data/Plan Information"
    PLAN_PARAMS_PATH = "Plan Data/Plan Parameters"
    PRECIP_PATH = "Event Conditions/Meteorology/Precipitation"
    RESULTS_UNSTEADY_PATH = "Results/Unsteady"
    RESULTS_UNSTEADY_SUMMARY_PATH = f"{RESULTS_UNSTEADY_PATH}/Summary"
    VOLUME_ACCOUNTING_PATH = f"{RESULTS_UNSTEADY_PATH}/Volume Accounting"
    BASE_OUTPUT_PATH = f"{RESULTS_UNSTEADY_PATH}/Output/Output Blocks/Base Output"
    SUMMARY_OUTPUT_2D_FLOW_AREAS_PATH = (
        f"{BASE_OUTPUT_PATH}/Summary Output/2D Flow Areas"
    )
    UNSTEADY_TIME_SERIES_PATH = f"{BASE_OUTPUT_PATH}/Unsteady Time Series"

    RESULTS_STEADY_PATH = "Results/Steady"
    BASE_STEADY_PATH = f"{RESULTS_STEADY_PATH}/Output/Output Blocks/Base Output"
    STEADY_PROFILES_PATH = f"{BASE_STEADY_PATH}/Steady Profiles"
    STEADY_XS_PATH = f"{STEADY_PROFILES_PATH}/Cross Sections"
    STEADY_XS_ADDITIONAL_PATH = f"{STEADY_XS_PATH}/Additional Variables"

    def __init__(self, name: str, **kwargs):
        """Open a HEC-RAS Plan HDF file.

        Parameters
        ----------
        name : str
            The path to the RAS Plan HDF file.
        kwargs : dict
            Additional keyword arguments to pass to h5py.File
        """
        super().__init__(name, **kwargs)

    def _simulation_start_time(self) -> datetime:
        """Return the simulation start time from the plan file.

        Returns
        -------
        datetime
            The simulation start time.
        """
        plan_info = self.get_plan_info_attrs()
        return plan_info["Simulation Start Time"]

    def _2d_flow_area_names_and_counts(self) -> List[Tuple[str, int]]:
        """
        Return a list of 2D flow area names and cell counts.

        Returns
        -------
        List[Tuple[str, int]]
            A list of tuples, where each tuple contains a 2D flow area name
            and the number of cells in that area.
        """
        d2_flow_areas = self[f"{self.FLOW_AREA_2D_PATH}/Attributes"][:]
        return [
            (d2_flow_area[0].decode("utf-8"), d2_flow_area[-1])
            for d2_flow_area in d2_flow_areas
        ]

    def _mesh_summary_output_group(
        self, mesh_name: str, output_var: SummaryOutputVar
    ) -> h5py.Group:
        """Return the HDF group for a 2D flow area summary output variable.

        Parameters
        ----------
        mesh_name : str
            The name of the 2D flow area mesh.
        output_var : str
            The name of the output variable.

        Returns
        -------
        h5py.Group
            The HDF group for the output variable.
        """
        output_path = (
            f"{self.SUMMARY_OUTPUT_2D_FLOW_AREAS_PATH}/{mesh_name}/{output_var.value}"
        )
        output_group = self.get(output_path)
        if output_group is None:
            raise RasPlanHdfError(
                f"Could not find HDF group at path '{output_path}'."
                " Does the Plan HDF file contain 2D output data?"
            )
        return output_group

    def _mesh_summary_output_min_max_values(
        self, mesh_name: str, var: SummaryOutputVar
    ) -> np.ndarray:
        """Return values for a "Maximum"/"Minimum" summary output variable.

        Parameters
        ----------
        mesh_name : str
            The name of the 2D flow area mesh.
        var : SummaryOutputVar
            The summary output variable to retrieve.

        Returns
        -------
        np.ndarray
            An array of maximum water surface elevation values.
        """
        max_ws_group = self._mesh_summary_output_group(mesh_name, var)
        max_ws_raw = max_ws_group[:]
        max_ws_values = max_ws_raw[0]
        return max_ws_values

    def _summary_output_min_max_time_unit(self, dataset: h5py.Dataset) -> str:
        """Return the time unit for "Maximum"/"Minimum" summary output datasets.

        I.e., for summary output such as "Maximum Water Surface", "Minimum Water Surface", etc.

        Should normally return the string: "days".

        Parameters
        ----------
        mesh_name : str
            The name of the 2D flow area mesh.

        Returns
        -------
        str
            The time unit for the maximum water surface elevation data.
        """
        if "Units per row" in dataset.attrs:
            units = dataset.attrs["Units per row"]
        else:
            units = dataset.attrs["Units"]
        # expect an array of size 2, with the first element being length or velocity units
        # and the second element being time units (e.g., ["ft", "days"])
        time_unit = units[1]
        return time_unit.decode("utf-8")

    def _mesh_summary_output_min_max_times(
        self,
        mesh_name: str,
        var: SummaryOutputVar,
        time_unit: str = "days",
        round_to: str = "0.1 s",
    ) -> np.ndarray[np.datetime64]:
        """Return an array of times for min/max summary output data.

        Parameters
        ----------
        mesh_name : str
            The name of the 2D flow area mesh.
        var : SummaryOutputVar
            The summary output variable to retrieve.
        time_unit : str, optional
            The time unit for the maximum water surface elevation data.
            Default: "days".
        round_to : str, optional
            The time unit to round the datetimes to. Default: "0.1 s" (seconds).

        Returns
        -------
        np.ndarray[np.datetime64]
            An array of times for the maximum water surface elevation data.
        """
        start_time = self._simulation_start_time()
        max_ws_group = self._mesh_summary_output_group(mesh_name, var)
        time_unit = self._summary_output_min_max_time_unit(max_ws_group)
        max_ws_raw = max_ws_group[:]
        max_ws_times_raw = max_ws_raw[1]
        # we get weirdly specific datetime values if we don't round to e.g., 0.1 seconds;
        # otherwise datetimes don't align with the actual timestep values in the plan file
        max_ws_times = ras_timesteps_to_datetimes(
            max_ws_times_raw, start_time, time_unit=time_unit, round_to=round_to
        )
        return max_ws_times

    def _mesh_summary_output_min_max(
        self,
        var: SummaryOutputVar,
        value_col: str = "value",
        time_col: str = "time",
        round_to: str = "0.1 s",
    ) -> DataFrame:
        """Return the min/max values and times for a summary output variable.

        Valid for:
        - Maximum Water Surface
        - Minimum Water Surface
        - Maximum Face Velocity
        - Minimum Face Velocity
        - Cell Maximum Water Surface Error

        Parameters
        ----------
        var : SummaryOutputVar
            The summary output variable to retrieve.
        round_to : str, optional
            The time unit to round the datetimes to. Default: "0.1 s" (seconds).

        Returns
        -------
        DataFrame
            A DataFrame with columns 'mesh_name', 'cell_id' or 'face_id', 'value', and 'time'.
        """
        dfs = []
        for mesh_name, cell_count in self._2d_flow_area_names_and_counts():
            values = self._mesh_summary_output_min_max_values(mesh_name, var)
            times = self._mesh_summary_output_min_max_times(
                mesh_name, var, round_to=round_to
            )
            if var in [
                SummaryOutputVar.MAXIMUM_FACE_VELOCITY,
                SummaryOutputVar.MINIMUM_FACE_VELOCITY,
            ]:
                geom_id_col = "face_id"
            else:
                geom_id_col = "cell_id"
                # The 2D mesh output data contains values for more cells than are actually
                # in the mesh. The the true number of cells for a mesh is found in the table:
                # "/Geometry/2D Flow Areas/Attributes". The number of cells in the 2D output
                # data instead matches the number of cells in the "Cells Center Coordinate"
                # array, which contains extra points along the perimeter of the mesh. These
                # extra points are appended to the end of the mesh data and contain bogus
                # output values (e.g., 0.0, NaN). We need to filter out these bogus values.
                values = values[:cell_count]
                times = times[:cell_count]
            df = DataFrame(
                {
                    "mesh_name": [mesh_name] * len(values),
                    geom_id_col: range(len(values)),
                    value_col: values,
                    time_col: times,
                }
            )
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        return df

    def _mesh_summary_output_basic(
        self, var: SummaryOutputVar, value_col: str = "value"
    ) -> DataFrame:
        """Return values and times for a summary output variable.

        Valid for:
        - Cell Cumulative Iteration (i.e. Cumulative Max Iterations)
        - Cell Last Iteration

        Parameters
        ----------
        var : SummaryOutputVar
            The summary output variable to retrieve.

        Returns
        -------
        DataFrame
            A DataFrame with columns 'mesh_name', 'cell_id' or 'face_id', 'value', and 'time'.
        """
        dfs = []
        for mesh_name, cell_count in self._2d_flow_area_names_and_counts():
            group = self._mesh_summary_output_group(mesh_name, var)
            values = group[:][:cell_count]
            df = DataFrame(
                {
                    "mesh_name": [mesh_name] * len(values),
                    "cell_id": range(len(values)),
                    value_col: values,
                }
            )
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        return df

    def mesh_max_iter(self) -> DataFrame:
        """Return the number of times each cell in the mesh reached the max number of iterations.

        Returns
        -------
        DataFrame
            A DataFrame with columns 'mesh_name', 'cell_id', and 'max_iterations'.
        """
        df = self._mesh_summary_output_basic(
            SummaryOutputVar.CELL_CUMULATIVE_ITERATION, value_col="max_iter"
        )
        return df

    def mesh_last_iter(self) -> DataFrame:
        """Return the number of times each cell in the mesh was the last cell to converge.

        Returns
        -------
        DataFrame
            A DataFrame with columns 'mesh_name', 'cell_id', and 'last_iter'.
        """
        df = self._mesh_summary_output_basic(
            SummaryOutputVar.CELL_LAST_ITERATION, value_col="last_iter"
        )
        return df

    def mesh_max_ws(self, round_to: str = "0.1 s") -> DataFrame:
        """Return the max water surface elevation for each cell in the mesh.

        Includes the corresponding time of max water surface elevation.

        Parameters
        ----------
        round_to : str, optional
            The time unit to round the datetimes to. Default: "0.1 s" (seconds).
            See Pandas documentation for valid time units:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        Returns
        -------
        DataFrame
            A DataFrame with columns 'mesh_name', 'cell_id', 'max_ws', and 'max_ws_time'.
        """
        df = self._mesh_summary_output_min_max(
            SummaryOutputVar.MAXIMUM_WATER_SURFACE,
            value_col="max_ws",
            time_col="max_ws_time",
            round_to=round_to,
        )
        return df

    def mesh_min_ws(self, round_to: str = "0.1 s") -> DataFrame:
        """Return the min water surface elevation for each cell in the mesh.

        Includes the corresponding time of min water surface elevation.

        Parameters
        ----------
        round_to : str, optional
            The time unit to round the datetimes to. Default: "0.1 s" (seconds).
            See Pandas documentation for valid time units:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

        Returns
        -------
        DataFrame
            A DataFrame with columns 'mesh_name', 'cell_id', 'min_ws', and 'min_ws_time'.
        """
        df = self._mesh_summary_output_min_max(
            SummaryOutputVar.MINIMUM_WATER_SURFACE,
            value_col="min_ws",
            time_col="min_ws_time",
            round_to=round_to,
        )
        return df

    def mesh_max_face_v(self, round_to: str = "0.1 s") -> DataFrame:
        """Return the max face velocity for each face in the mesh.

        Includes the corresponding time of max face velocity.

        Parameters
        ----------
        round_to : str, optional
            The time unit to round the datetimes to. Default: "0.1 s" (seconds).
            See Pandas documentation for valid time units:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

        Returns
        -------
        DataFrame
            A DataFrame with columns 'mesh_name', 'face_id', 'max_v', and 'max_v_time'.
        """
        df = self._mesh_summary_output_min_max(
            SummaryOutputVar.MAXIMUM_FACE_VELOCITY,
            value_col="max_v",
            time_col="max_v_time",
            round_to=round_to,
        )
        return df

    def mesh_min_face_v(self, round_to: str = "0.1 s") -> DataFrame:
        """Return the min face velocity for each face in the mesh.

        Includes the corresponding time of min face velocity.

        Parameters
        ----------
        round_to : str, optional
            The time unit to round the datetimes to. Default: "0.1 s" (seconds).
            See Pandas documentation for valid time units:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

        Returns
        -------
        DataFrame
            A DataFrame with columns 'mesh_name', 'face_id', 'min_v', and 'min_v_time'.
        """
        df = self._mesh_summary_output_min_max(
            SummaryOutputVar.MINIMUM_FACE_VELOCITY,
            value_col="min_v",
            time_col="min_v_time",
            round_to=round_to,
        )
        return df

    def mesh_max_ws_err(self, round_to: str = "0.1 s") -> DataFrame:
        """Return the max water surface error for each cell in the mesh.

        Includes the corresponding time of max water surface error.

        Parameters
        ----------
        round_to : str, optional
            The time unit to round the datetimes to. Default: "0.1 s" (seconds).
            See Pandas documentation for valid time units:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

        Returns
        -------
        DataFrame
            A DataFrame with columns 'mesh_name', 'cell_id', 'max_ws_err', and 'max_ws_err_time'.
        """
        df = self._mesh_summary_output_min_max(
            SummaryOutputVar.CELL_MAXIMUM_WATER_SURFACE_ERROR,
            value_col="max_ws_err",
            time_col="max_ws_err_time",
            round_to=round_to,
        )
        return df

    def mesh_summary_output(
        self, var: SummaryOutputVar, round_to: str = "0.1 s"
    ) -> DataFrame:
        """Return the summary output data for a given variable.

        Parameters
        ----------
        var : SummaryOutputVar
            The summary output variable to retrieve.

        Returns
        -------
        DataFrame
            A DataFrame with columns 'mesh_name', 'cell_id' or 'face_id', a value column, and a time column.
        """
        methods_with_times = {
            SummaryOutputVar.MAXIMUM_WATER_SURFACE: self.mesh_max_ws,
            SummaryOutputVar.MINIMUM_WATER_SURFACE: self.mesh_min_ws,
            SummaryOutputVar.MAXIMUM_FACE_VELOCITY: self.mesh_max_face_v,
            SummaryOutputVar.MINIMUM_FACE_VELOCITY: self.mesh_min_face_v,
            SummaryOutputVar.CELL_MAXIMUM_WATER_SURFACE_ERROR: self.mesh_max_ws_err,
        }
        other_methods = {
            SummaryOutputVar.CELL_CUMULATIVE_ITERATION: self.mesh_max_iter,
            SummaryOutputVar.CELL_LAST_ITERATION: self.mesh_last_iter,
        }
        if var in methods_with_times:
            df = methods_with_times[var](round_to=round_to)
        else:
            df = other_methods[var]()
        return df

    def _summary_output_vars(
        self, cells_or_faces: Optional[str] = None
    ) -> List[SummaryOutputVar]:
        """Return a list of available summary output variables from the Plan HDF file.

        Returns
        -------
        List[SummaryOutputVar]
            A list of summary output variables.
        """
        mesh_names_counts = self._2d_flow_area_names_and_counts()
        mesh_names = [mesh_name for mesh_name, _ in mesh_names_counts]
        vars = set()
        for mesh_name in mesh_names:
            path = f"{self.SUMMARY_OUTPUT_2D_FLOW_AREAS_PATH}/{mesh_name}"
            datasets = self[path].keys()
            for dataset in datasets:
                try:
                    var = SummaryOutputVar(dataset)
                except ValueError:
                    continue
                vars.add(var)
        if cells_or_faces == "cells":
            vars = vars.intersection(SUMMARY_OUTPUT_VARS_CELLS)
        elif cells_or_faces == "faces":
            vars = vars.intersection(SUMMARY_OUTPUT_VARS_FACES)
        return sorted(list(vars), key=lambda x: x.value)

    def _mesh_summary_outputs_gdf(
        self,
        geom_func: str,
        cells_or_faces: str = "cells",
        include_output: Union[bool, List[SummaryOutputVar]] = True,
        round_to: str = "0.1 s",
        datetime_to_str: bool = False,
    ) -> GeoDataFrame:
        """Return a GeoDataFrame with mesh geometry and summary output data.

        Parameters
        ----------
        geom_func : str
            The method name to call to get the mesh geometry.
        cells_or_faces : str, optional
            The type of geometry to include in the GeoDataFrame.
            Must be either "cells" or "faces". (default: "cells")
        include_output : Union[bool, List[SummaryOutputVar]], optional
            If True, include all available summary output data in the GeoDataFrame.
            If a list of SummaryOutputVar values, include only the specified summary output data.
            If False, do not include any summary output data.
            (default: True)
        round_to : str, optional
            The time unit to round the datetimes to. Default: "0.1 s" (seconds).
        datetime_to_str : bool, optional
            If True, convert datetime columns to strings. (default: False)
        """
        gdf = getattr(super(), geom_func)()
        if include_output is False:
            return gdf
        if include_output is True:
            summary_output_vars = self._summary_output_vars(
                cells_or_faces=cells_or_faces
            )
        elif isinstance(include_output, list):
            summary_output_vars = []
            for var in include_output:
                if not isinstance(var, SummaryOutputVar):
                    var = SummaryOutputVar(var)
                summary_output_vars.append(var)
        else:
            raise ValueError(
                "include_output must be a boolean or a list of SummaryOutputVar values."
            )
        if cells_or_faces == "cells":
            geom_id_col = "cell_id"
        elif cells_or_faces == "faces":
            geom_id_col = "face_id"
        else:
            raise ValueError('cells_or_faces must be either "cells" or "faces".')
        for var in summary_output_vars:
            df = self.mesh_summary_output(var, round_to=round_to)
            gdf = gdf.merge(df, on=["mesh_name", geom_id_col], how="left")
        if datetime_to_str:
            gdf = df_datetimes_to_str(gdf)
        return gdf

    def mesh_cell_points(
        self,
        include_output: Union[bool, List[SummaryOutputVar], List[str]] = True,
        round_to: str = "0.1 s",
        datetime_to_str: bool = False,
    ) -> GeoDataFrame:
        """Return the cell points for each cell in the mesh, including summary output.

        Parameters
        ----------
        include_output : Union[bool, List[SummaryOutputVar], List[str]], optional
            If True, include all available summary output data in the GeoDataFrame.
            If a list of SummaryOutputVar values, include only the specified summary output data.
            If a list of strings, include only the specified summary output data by name.
            If False, do not include any summary output data.
            (default: True)
        round_to : str, optional
            The time unit to round the datetimes to. Default: "0.1 s" (seconds).
        datetime_to_str : bool, optional
            If True, convert datetime columns to strings. (default: False)

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame with columns 'mesh_name', 'cell_id', 'geometry', and columns for each
            summary output variable.
        """
        return self._mesh_summary_outputs_gdf(
            "mesh_cell_points",
            "cells",
            include_output=include_output,
            round_to=round_to,
            datetime_to_str=datetime_to_str,
        )

    def mesh_cell_polygons(
        self,
        include_output: Union[bool, List[SummaryOutputVar], List[str]] = True,
        round_to: str = "0.1 s",
        datetime_to_str: bool = False,
    ) -> GeoDataFrame:
        """Return the cell polygons for each cell in the mesh, including output.

        Parameters
        ----------
        include_output : Union[bool, List[SummaryOutputVar], List[str]], optional
            If True, include all available summary output data in the GeoDataFrame.
            If a list of SummaryOutputVar values, include only the specified summary output data.
            If a list of strings, include only the specified summary output data by name.
            If False, do not include any summary output data.
            (default: True)
        round_to : str, optional
            The time unit to round the datetimes to. Default: "0.1 s" (seconds).
        datetime_to_str : bool, optional
            If True, convert datetime columns to strings. (default: False)

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame with columns 'mesh_name', 'cell_id', 'geometry', and columns for each
            summary output variable.
        """
        return self._mesh_summary_outputs_gdf(
            "mesh_cell_polygons",
            "cells",
            include_output=include_output,
            round_to=round_to,
            datetime_to_str=datetime_to_str,
        )

    def mesh_cell_faces(
        self,
        include_output: Union[bool, List[SummaryOutputVar], List[str]] = True,
        round_to: str = "0.1 s",
        datetime_to_str: bool = False,
    ) -> GeoDataFrame:
        """Return the cell faces for each cell in the mesh, including output.

        Parameters
        ----------
        include_output : Union[bool, List[SummaryOutputVar], List[str]], optional
            If True, include all available summary output data in the GeoDataFrame.
            If a list of SummaryOutputVar values, include only the specified summary output data.
            If a list of strings, include only the specified summary output data by name.
            If False, do not include any summary output data.
            (default: True)
        round_to : str, optional
            The time unit to round the datetimes to. Default: "0.1 s" (seconds).
        datetime_to_str : bool, optional
            If True, convert datetime columns to strings. (default: False)

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame with columns 'mesh_name', 'cell_id', 'geometry', and columns for each
            summary output variable.
        """
        return self._mesh_summary_outputs_gdf(
            "mesh_cell_faces",
            "faces",
            include_output=include_output,
            round_to=round_to,
            datetime_to_str=datetime_to_str,
        )

    def unsteady_datetimes(self) -> List[datetime]:
        """Return the unsteady timeseries datetimes from the plan file.

        Returns
        -------
        List[datetime]
            A list of datetimes for the unsteady timeseries data.
        """
        group_path = f"{self.UNSTEADY_TIME_SERIES_PATH}/Time Date Stamp (ms)"
        raw_datetimes = self[group_path][:]
        dt = [parse_ras_datetime_ms(x.decode("utf-8")) for x in raw_datetimes]
        return dt

    def _mesh_timeseries_output_values_units(
        self,
        mesh_name: str,
        var: TimeSeriesOutputVar,
    ) -> Tuple[np.ndarray, str]:
        path = f"{self.UNSTEADY_TIME_SERIES_PATH}/2D Flow Areas/{mesh_name}/{var.value}"
        group = self.get(path)
        try:
            import dask.array as da

            # TODO: user-specified chunks?
            values = da.from_array(group, chunks=group.chunks)
        except ImportError:
            values = group[:]
        units = group.attrs.get("Units")
        if units is not None:
            units = units.decode("utf-8")
        return values, units

    def mesh_timeseries_output(
        self,
        mesh_name: str,
        var: Union[str, TimeSeriesOutputVar],
    ) -> xr.DataArray:
        """Return the time series output data for a given variable.

        Parameters
        ----------
        mesh_name : str
            The name of the 2D flow area mesh.
        var : TimeSeriesOutputVar
            The time series output variable to retrieve.

        Returns
        -------
        xr.DataArray
            An xarray DataArray with dimensions 'time' and 'cell_id'.
        """
        times = self.unsteady_datetimes()
        mesh_names_counts = {
            name: count for name, count in self._2d_flow_area_names_and_counts()
        }
        if mesh_name not in mesh_names_counts:
            raise ValueError(f"Mesh '{mesh_name}' not found in the Plan HDF file.")
        if isinstance(var, str):
            var = TimeSeriesOutputVar(var)
        values, units = self._mesh_timeseries_output_values_units(mesh_name, var)
        if var in TIME_SERIES_OUTPUT_VARS_CELLS:
            cell_count = mesh_names_counts[mesh_name]
            values = values[:, :cell_count]
            id_coord = "cell_id"
        elif var in TIME_SERIES_OUTPUT_VARS_FACES:
            id_coord = "face_id"
        else:
            raise ValueError(f"Invalid time series output variable: {var.value}")
        da = xr.DataArray(
            values,
            name=var.value,
            dims=["time", id_coord],
            coords={
                "time": times,
                id_coord: range(values.shape[1]),
            },
            attrs={
                "mesh_name": mesh_name,
                "variable": var.value,
                "units": units,
            },
        )
        return da

    def _mesh_timeseries_outputs(
        self, mesh_name: str, vars: List[TimeSeriesOutputVar]
    ) -> xr.Dataset:
        datasets = {}
        for var in vars:
            var_path = f"{self.UNSTEADY_TIME_SERIES_PATH}/2D Flow Areas/{mesh_name}/{var.value}"
            if self.get(var_path) is None:
                continue
            da = self.mesh_timeseries_output(mesh_name, var)
            datasets[var.value] = da
        ds = xr.Dataset(datasets, attrs={"mesh_name": mesh_name})
        return ds

    def mesh_timeseries_output_cells(self, mesh_name: str) -> xr.Dataset:
        """Return the time series output data for cells in a 2D flow area mesh.

        Parameters
        ----------
        mesh_name : str
            The name of the 2D flow area mesh.

        Returns
        -------
        xr.Dataset
            An xarray Dataset with DataArrays for each time series output variable.
        """
        ds = self._mesh_timeseries_outputs(mesh_name, TIME_SERIES_OUTPUT_VARS_CELLS)
        return ds

    def mesh_timeseries_output_faces(self, mesh_name: str) -> xr.Dataset:
        """Return the time series output data for faces in a 2D flow area mesh.

        Parameters
        ----------
        mesh_name : str
            The name of the 2D flow area mesh.

        Returns
        -------
        xr.Dataset
            An xarray Dataset with DataArrays for each time series output variable.
        """
        ds = self._mesh_timeseries_outputs(mesh_name, TIME_SERIES_OUTPUT_VARS_FACES)
        return ds

    def get_plan_info_attrs(self) -> Dict:
        """Return plan information attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary of plan information attributes.
        """
        return self.get_attrs(self.PLAN_INFO_PATH)

    def get_plan_param_attrs(self) -> Dict:
        """Return plan parameter attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary of plan parameter attributes.
        """
        return self.get_attrs(self.PLAN_PARAMS_PATH)

    def get_meteorology_precip_attrs(self) -> Dict:
        """Return precipitation attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary of precipitation attributes.
        """
        return self.get_attrs(self.PRECIP_PATH)

    def get_results_unsteady_attrs(self) -> Dict:
        """Return unsteady attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary of unsteady attributes.
        """
        return self.get_attrs(self.RESULTS_UNSTEADY_PATH)

    def get_results_unsteady_summary_attrs(self) -> Dict:
        """Return results unsteady summary attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary of results summary attributes.
        """
        return self.get_attrs(self.RESULTS_UNSTEADY_SUMMARY_PATH)

    def get_results_volume_accounting_attrs(self) -> Dict:
        """Return volume accounting attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary of volume accounting attributes.
        """
        return self.get_attrs(self.VOLUME_ACCOUNTING_PATH)

    def enroachment_points(self) -> GeoDataFrame:  # noqa: D102
        raise NotImplementedError

    def steady_flow_names(self) -> list:
        """Return the profile information for each steady flow event.

        Returns
        -------
        DataFrame
            A Dataframe containing the profile names for each event
        """
        if self.STEADY_PROFILES_PATH not in self:
            return pd.DataFrame()

        profile_data = self[self.STEADY_PROFILES_PATH]
        profile_attrs = profile_data["Profile Names"][()]

        return [x.decode("utf-8") for x in profile_attrs]

    def steady_profile_xs_output(
        self, var: XsSteadyOutputVar, round_to: int = 2
    ) -> DataFrame:
        """Create a Dataframe from steady cross section results based on path.

        Parameters
        ----------
        var : XsSteadyOutputVar
            The name of the table in the steady results that information is to be retireved from.

        round_to : int, optional
            Number of decimal places to round output data to.

        Returns
        -------
            Dataframe with desired hdf data.
        """
        if var in XS_STEADY_OUTPUT_ADDITIONAL:
            path = f"{self.STEADY_XS_ADDITIONAL_PATH}/{var.value}"
        else:
            path = f"{self.STEADY_XS_PATH}/{var.value}"
        if path not in self:
            return DataFrame()

        profiles = self.steady_flow_names()

        steady_data = self[path]
        df = DataFrame(steady_data, index=profiles)
        df_t = df.T.copy()
        for p in profiles:
            df_t[p] = df_t[p].apply(lambda x: round(x, round_to))

        return df_t

    def cross_sections_wsel(self) -> DataFrame:
        """Return the water surface elevation information for each 1D Cross Section.

        Returns
        -------
        DataFrame
            A Dataframe containing the water surface elevations for each cross section and event
        """
        return self.steady_profile_xs_output(XsSteadyOutputVar.WATER_SURFACE)

    def cross_sections_flow(self) -> DataFrame:
        """Return the Flow information for each 1D Cross Section.

        Returns
        -------
        DataFrame
            A Dataframe containing the flow for each cross section and event
        """
        return self.steady_profile_xs_output(XsSteadyOutputVar.FLOW)

    def cross_sections_energy_grade(self) -> DataFrame:
        """Return the energy grade information for each 1D Cross Section.

        Returns
        -------
        DataFrame
            A Dataframe containing the water surface elevations for each cross section and event
        """
        return self.steady_profile_xs_output(XsSteadyOutputVar.ENERGY_GRADE)

    def cross_sections_additional_enc_station_left(self) -> DataFrame:
        """Return the left side encroachment information for a floodway plan hdf.

        Returns
        -------
        DataFrame
            A DataFrame containing the cross sections left side encroachment stations
        """
        return self.steady_profile_xs_output(
            XsSteadyOutputVar.ENCROACHMENT_STATION_LEFT
        )

    def cross_sections_additional_enc_station_right(self) -> DataFrame:
        """Return the right side encroachment information for a floodway plan hdf.

        Returns
        -------
        DataFrame
            A DataFrame containing the cross sections right side encroachment stations
        """
        return self.steady_profile_xs_output(
            XsSteadyOutputVar.ENCROACHMENT_STATION_RIGHT
        )

    def cross_sections_additional_area_total(self) -> DataFrame:
        """Return the 1D cross section area for each profile.

        Returns
        -------
        DataFrame
            A DataFrame containing the wet area inside the cross sections
        """
        return self.steady_profile_xs_output(XsSteadyOutputVar.AREA_INEFFECTIVE_TOTAL)

    def cross_sections_additional_velocity_total(self) -> DataFrame:
        """Return the 1D cross section velocity for each profile.

        Returns
        -------
        DataFrame
            A DataFrame containing the velocity inside the cross sections
        """
        return self.steady_profile_xs_output(XsSteadyOutputVar.VELOCITY_TOTAL)
