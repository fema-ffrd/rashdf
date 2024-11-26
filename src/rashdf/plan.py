"""HEC-RAS Plan HDF class."""

from .geom import RasGeomHdf
from .utils import (
    df_datetimes_to_str,
    ras_timesteps_to_datetimes,
    parse_ras_datetime_ms,
    deprecated,
    convert_ras_hdf_value,
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
    OBS_DATA_PATH = "Event Conditions/Observed Data"
    RESULTS_UNSTEADY_PATH = "Results/Unsteady"
    RESULTS_UNSTEADY_SUMMARY_PATH = f"{RESULTS_UNSTEADY_PATH}/Summary"
    VOLUME_ACCOUNTING_PATH = f"{RESULTS_UNSTEADY_PATH}/Volume Accounting"
    BASE_OUTPUT_PATH = f"{RESULTS_UNSTEADY_PATH}/Output/Output Blocks/Base Output"
    SUMMARY_OUTPUT_2D_FLOW_AREAS_PATH = (
        f"{BASE_OUTPUT_PATH}/Summary Output/2D Flow Areas"
    )
    UNSTEADY_TIME_SERIES_PATH = f"{BASE_OUTPUT_PATH}/Unsteady Time Series"
    REFERENCE_LINES_OUTPUT_PATH = f"{UNSTEADY_TIME_SERIES_PATH}/Reference Lines"
    REFERENCE_POINTS_OUTPUT_PATH = f"{UNSTEADY_TIME_SERIES_PATH}/Reference Points"
    OBS_FLOW_OUTPUT_PATH = f"{OBS_DATA_PATH}/Flow"
    OBS_STAGE_OUTPUT_PATH = f"{OBS_DATA_PATH}/Stage"

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
            A DataFrame with columns 'mesh_name', 'cell_id' or 'face_id', a value column,
            and a time column if the value corresponds to a specific time.
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

    def _mesh_summary_outputs_df(
        self,
        cells_or_faces: str,
        output_vars: Optional[List[SummaryOutputVar]] = None,
        round_to: str = "0.1 s",
    ) -> DataFrame:
        if cells_or_faces == "cells":
            feature_id_field = "cell_id"
        elif cells_or_faces == "faces":
            feature_id_field = "face_id"
        else:
            raise ValueError('cells_or_faces must be either "cells" or "faces".')
        if output_vars is None:
            summary_output_vars = self._summary_output_vars(
                cells_or_faces=cells_or_faces
            )
        elif isinstance(output_vars, list):
            summary_output_vars = []
            for var in output_vars:
                if not isinstance(var, SummaryOutputVar):
                    var = SummaryOutputVar(var)
                summary_output_vars.append(var)
        else:
            raise ValueError(
                "include_output must be a boolean or a list of SummaryOutputVar values."
            )
        df = self.mesh_summary_output(summary_output_vars[0], round_to=round_to)
        for var in summary_output_vars[1:]:
            df_var = self.mesh_summary_output(var, round_to=round_to)
            df = df.merge(df_var, on=["mesh_name", feature_id_field], how="left")
        return df

    def mesh_cells_summary_output(self, round_to: str = "0.1 s") -> DataFrame:
        """
        Return a DataFrame with summary output data for each mesh cell in the model.

        Parameters
        ----------
        round_to : str, optional
            The time unit to round the datetimes to. Default: "0.1 s" (seconds).
            See Pandas documentation for valid time units:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

        Returns
        -------
        DataFrame
            A DataFrame with columns 'mesh_name', 'cell_id', and columns for each
            summary output variable.
        """
        return self._mesh_summary_outputs_df("cells", round_to=round_to)

    def mesh_faces_summary_output(self, round_to: str = "0.1 s") -> DataFrame:
        """
        Return a DataFrame with summary output data for each mesh face in the model.

        Parameters
        ----------
        round_to : str, optional
            The time unit to round the datetimes to. Default: "0.1 s" (seconds).
            See Pandas documentation for valid time units:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

        Returns
        -------
        DataFrame
            A DataFrame with columns 'mesh_name', 'face_id', and columns for each
            summary output variable.
        """
        return self._mesh_summary_outputs_df("faces", round_to=round_to)

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
        path = self._mesh_timeseries_output_path(mesh_name, var.value)
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
        truncate: bool = True,
    ) -> xr.DataArray:
        """Return the time series output data for a given variable.

        Parameters
        ----------
        mesh_name : str
            The name of the 2D flow area mesh.
        var : TimeSeriesOutputVar
            The time series output variable to retrieve.
        truncate : bool, optional
            If True, truncate the number of cells to the listed cell count.

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
            if truncate:
                values = values[:, :cell_count]
            else:
                values = values[:, :]
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
                "hdf_path": self._mesh_timeseries_output_path(mesh_name, var.value),
            },
        )
        return da

    def _mesh_timeseries_output_path(self, mesh_name: str, var_name: str) -> str:
        return f"{self.UNSTEADY_TIME_SERIES_PATH}/2D Flow Areas/{mesh_name}/{var_name}"

    def _mesh_timeseries_outputs(
        self, mesh_name: str, vars: List[TimeSeriesOutputVar], truncate: bool = True
    ) -> xr.Dataset:
        datasets = {}
        for var in vars:
            var_path = f"{self.UNSTEADY_TIME_SERIES_PATH}/2D Flow Areas/{mesh_name}/{var.value}"
            if self.get(var_path) is None:
                continue
            da = self.mesh_timeseries_output(mesh_name, var, truncate=truncate)
            datasets[var.value] = da
        ds = xr.Dataset(datasets, attrs={"mesh_name": mesh_name})
        return ds

    def mesh_cells_timeseries_output(self, mesh_name: str) -> xr.Dataset:
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

    @deprecated
    def mesh_timeseries_output_cells(self, mesh_name: str) -> xr.Dataset:
        """Return the time series output data for cells in a 2D flow area mesh.

        Deprecated: use mesh_cells_timeseries_output instead.

        Parameters
        ----------
        mesh_name : str
            The name of the 2D flow area mesh.

        Returns
        -------
        xr.Dataset
            An xarray Dataset with DataArrays for each time series output variable.
        """
        return self.mesh_cells_timeseries_output(mesh_name)

    def mesh_faces_timeseries_output(self, mesh_name: str) -> xr.Dataset:
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

    @deprecated
    def mesh_timeseries_output_faces(self, mesh_name: str) -> xr.Dataset:
        """Return the time series output data for faces in a 2D flow area mesh.

        Deprecated: use mesh_faces_timeseries_output instead.

        Parameters
        ----------
        mesh_name : str
            The name of the 2D flow area mesh.

        Returns
        -------
        xr.Dataset
            An xarray Dataset with DataArrays for each time series output variable.
        """
        return self.mesh_faces_timeseries_output(mesh_name)

    def reference_timeseries_output(self, reftype: str = "lines") -> xr.Dataset:
        """Return timeseries output data for reference lines or points from a HEC-RAS HDF plan file.

        Parameters
        ----------
        reftype : str, optional
            The type of reference data to retrieve. Must be either "lines" or "points".
            (default: "lines")

        Returns
        -------
        xr.Dataset
            An xarray Dataset with reference line timeseries data.
        """
        if reftype == "lines":
            output_path = self.REFERENCE_LINES_OUTPUT_PATH
            abbrev = "refln"
        elif reftype == "points":
            output_path = self.REFERENCE_POINTS_OUTPUT_PATH
            abbrev = "refpt"
        else:
            raise ValueError('reftype must be either "lines" or "points".')
        reference_group = self.get(output_path)
        if reference_group is None:
            raise RasPlanHdfError(
                f"Could not find HDF group at path '{output_path}'."
                f" Does the Plan HDF file contain reference {reftype[:-1]} output data?"
            )
        reference_names = reference_group["Name"][:]
        names = []
        mesh_areas = []
        for s in reference_names:
            name, mesh_area = s.decode("utf-8").split("|")
            names.append(name)
            mesh_areas.append(mesh_area)

        times = self.unsteady_datetimes()

        das = {}
        for var in ["Flow", "Velocity", "Water Surface"]:
            group = reference_group.get(var)
            if group is None:
                continue
            try:
                import dask.array as da

                # TODO: user-specified chunks?
                values = da.from_array(group, chunks=group.chunks)
            except ImportError:
                values = group[:]
            units = group.attrs["Units"].decode("utf-8")
            da = xr.DataArray(
                values,
                name=var,
                dims=["time", f"{abbrev}_id"],
                coords={
                    "time": times,
                    f"{abbrev}_id": range(values.shape[1]),
                    f"{abbrev}_name": (f"{abbrev}_id", names),
                    "mesh_name": (f"{abbrev}_id", mesh_areas),
                },
                attrs={"units": units, "hdf_path": f"{output_path}/{var}"},
            )
            das[var] = da
        return xr.Dataset(das)

    def reference_lines_timeseries_output(self) -> xr.Dataset:
        """Return timeseries output data for reference lines from a HEC-RAS HDF plan file.

        Returns
        -------
        xr.Dataset
            An xarray Dataset with timeseries output data for reference lines.
        """
        return self.reference_timeseries_output(reftype="lines")

    def observed_timeseries_input(self, vartype: str = "Flow") -> xr.DataArray:
        """Return observed timeseries input data for reference lines and points from a HEC-RAS HDF plan file.

        Parameters
        ----------
        vartype : str, optional
            The type of observed data to retrieve. Must be either "Flow" or "Stage".
            (default: "Flow")

        Returns
        -------
        xr.DataArray
            An xarray DataArray with observed timeseries input data for reference lines or reference points.
        """
        if vartype == "Flow":
            output_path = self.OBS_FLOW_OUTPUT_PATH
        elif vartype == "Stage":
            output_path = self.OBS_STAGE_OUTPUT_PATH
        else:
            raise ValueError('vartype must be either "Flow" or "Stage".')

        observed_group = self.get(output_path)
        if observed_group is None:
            raise RasPlanHdfError(
                f"Could not find HDF group at path '{output_path}'."
                f" Does the Plan HDF file contain reference {vartype} output data?"
            )
        if "Attributes" in observed_group.keys():
            attr_path = observed_group["Attributes"]
            attrs_df = pd.DataFrame(attr_path[:]).map(convert_ras_hdf_value)

        das = {}
        for idx, site in enumerate(observed_group.keys()):
            if site != "Attributes":
                # Site Ex: 'Ref Point: Grapevine_Lake_RP'
                site_path = observed_group[site]
                site_name = site.split(":")[1][1:]  # Grapevine_Lake_RP
                ref_type = site.split(":")[0]  # Ref Point
                if ref_type == "Ref Line":
                    ref_type = "refln"
                else:
                    ref_type = "refpt"
                df = pd.DataFrame(site_path[:]).map(convert_ras_hdf_value)
                # rename Date to time
                df = df.rename(columns={"Date": "time"})
                # Ensure the Date index is unique
                df = df.drop_duplicates(subset="time")
                # Package into an 1D xarray DataArray
                values = df["Value"].values
                times = df["time"].values
                da = xr.DataArray(
                    values,
                    name=vartype,
                    dims=["time"],
                    coords={
                        "time": times,
                    },
                    attrs={
                        "hdf_path": f"{output_path}/{site}",
                    },
                )
                # Expand dimensions to add additional coordinates
                da = da.expand_dims({f"{ref_type}_id": [idx - 1]})
                da = da.expand_dims({f"{ref_type}_name": [site_name]})
                das[site_name] = da
        das = xr.concat([das[site] for site in das.keys()], dim="time")
        return das

    def reference_points_timeseries_output(self) -> xr.Dataset:
        """Return timeseries output data for reference points from a HEC-RAS HDF plan file.

        Returns
        -------
        xr.Dataset
            An xarray Dataset with timeseries output data for reference points.
        """
        return self.reference_timeseries_output(reftype="points")

    def reference_summary_output(self, reftype: str = "lines") -> DataFrame:
        """Return summary output data for reference lines or points from a HEC-RAS HDF plan file.

        Returns
        -------
        DataFrame
            A DataFrame with reference line summary output data.
        """
        if reftype == "lines":
            abbrev = "refln"
        elif reftype == "points":
            abbrev = "refpt"
        else:
            raise ValueError('reftype must be either "lines" or "points".')
        ds = self.reference_timeseries_output(reftype=reftype)
        result = {
            f"{abbrev}_id": ds[f"{abbrev}_id"],
            f"{abbrev}_name": ds[f"{abbrev}_name"],
            "mesh_name": ds.mesh_name,
        }
        vars = {
            "Flow": "q",
            "Water Surface": "ws",
            "Velocity": "v",
        }
        for var, abbrev in vars.items():
            if var not in ds:
                continue
            max_var = ds[var].max(dim="time")
            max_time = ds[var].time[ds[var].argmax(dim="time")]
            min_var = ds[var].min(dim="time")
            min_time = ds[var].time[ds[var].argmin(dim="time")]
            result[f"max_{abbrev}"] = max_var
            result[f"max_{abbrev}_time"] = max_time
            result[f"min_{abbrev}"] = min_var
            result[f"min_{abbrev}_time"] = min_time
        return DataFrame(result)

    def _reference_lines_points(
        self,
        reftype: str = "lines",
        include_output: bool = True,
        datetime_to_str: bool = False,
    ) -> GeoDataFrame:
        if reftype == "lines":
            abbrev = "refln"
            gdf = super().reference_lines()
        elif reftype == "points":
            abbrev = "refpt"
            gdf = super().reference_points()
        else:
            raise ValueError('reftype must be either "lines" or "points".')
        if include_output is False:
            return gdf
        summary_output = self.reference_summary_output(reftype=reftype)
        gdf = gdf.merge(
            summary_output,
            on=[f"{abbrev}_id", f"{abbrev}_name", "mesh_name"],
            how="left",
        )
        if datetime_to_str:
            gdf = df_datetimes_to_str(gdf)
        return gdf

    def reference_lines(
        self, include_output: bool = True, datetime_to_str: bool = False
    ) -> GeoDataFrame:
        """Return the reference lines from a HEC-RAS HDF plan file.

        Includes summary output data for each reference line:
        - Maximum flow & time (max_q, max_q_time)
        - Minimum flow & time (min_q, min_q_time)
        - Maximum water surface elevation & time (max_ws, max_ws_time)
        - Minimum water surface elevation & time (min_ws, min_ws_time)

        Parameters
        ----------
        include_output : bool, optional
            If True, include summary output data in the GeoDataFrame. (default: True)
        datetime_to_str : bool, optional
            If True, convert datetime columns to strings. (default: False)

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame with reference line geometry and summary output data.
        """
        return self._reference_lines_points(
            reftype="lines",
            include_output=include_output,
            datetime_to_str=datetime_to_str,
        )

    def reference_points(
        self, include_output: bool = True, datetime_to_str: bool = False
    ) -> GeoDataFrame:
        """Return the reference points from a HEC-RAS HDF plan file.

        Parameters
        ----------
        include_output : bool, optional
            If True, include summary output data in the GeoDataFrame. (default: True)
        datetime_to_str : bool, optional
            If True, convert datetime columns to strings. (default: False)

        Includes summary output data for each reference point:
        - Maximum flow & time (max_q, max_q_time)
        - Minimum flow & time (min_q, min_q_time)
        - Maximum water surface elevation & time (max_ws, max_ws_time)
        - Minimum water surface elevation & time (min_ws, min_ws_time)

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame with reference point geometry and summary output data.
        """
        return self._reference_lines_points(
            reftype="points",
            include_output=include_output,
            datetime_to_str=datetime_to_str,
        )

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

    def _zmeta(self, ds: xr.Dataset) -> Dict:
        """Given a xarray Dataset, return kerchunk-style zarr reference metadata."""
        from kerchunk.hdf import SingleHdf5ToZarr
        import zarr
        import base64

        encoding = {}
        chunk_meta = {}

        # Loop through each variable / DataArray in the Dataset
        for var, da in ds.data_vars.items():
            # The "hdf_path" attribute is the path within the HDF5 file
            # that the DataArray was read from. This is attribute is inserted
            # by rashdf (see "mesh_timeseries_output" method).
            hdf_ds_path = da.attrs["hdf_path"]
            hdf_ds = self.get(hdf_ds_path)
            if hdf_ds is None:
                # If we don't know where in the HDF5 the data came from, we
                # have to skip it, because we won't be able to generate the
                # correct metadata for it.
                continue
            # Get the filters and storage info for the HDF5 dataset.
            # Calling private methods from Kerchunk here because
            # there's not a nice public API for this part. This is hacky
            # and a bit risky because these private methods are more likely
            # to change, but short of reimplementing these functions ourselves
            # it's the best way to get the metadata we need.
            # TODO: raise an issue in Kerchunk to expose this functionality?
            filters = SingleHdf5ToZarr._decode_filters(None, hdf_ds)
            encoding[var] = {"compressor": None, "filters": filters}
            storage_info = SingleHdf5ToZarr._storage_info(None, hdf_ds)
            # Generate chunk metadata for the DataArray
            for key, value in storage_info.items():
                chunk_number = ".".join([str(k) for k in key])
                chunk_key = f"{var}/{chunk_number}"
                chunk_meta[chunk_key] = [str(self._loc), value["offset"], value["size"]]
        # "Write" the Dataset to a temporary in-memory zarr store (which
        # is the same a Python dictionary)
        zarr_tmp = zarr.MemoryStore()
        # Use compute=False here because we don't _actually_ want to write
        # the data to the zarr store, we just want to generate the metadata.
        ds.to_zarr(zarr_tmp, mode="w", compute=False, encoding=encoding)
        zarr_meta = {"version": 1, "refs": {}}
        # Loop through the in-memory Zarr store, decode the data to strings,
        # and add it to the final metadata dictionary.
        for key, value in zarr_tmp.items():
            try:
                value_str = value.decode("utf-8")
            except UnicodeDecodeError:
                value_str = "base64:" + base64.b64encode(value).decode("utf-8")
            zarr_meta["refs"][key] = value_str
        zarr_meta["refs"].update(chunk_meta)
        return zarr_meta

    def zmeta_mesh_cells_timeseries_output(self, mesh_name: str) -> Dict:
        """Return kerchunk-style zarr reference metadata.

        Requires the 'zarr' and 'kerchunk' packages.

        Returns
        -------
        dict
            Dictionary of kerchunk-style zarr reference metadata.
        """
        ds = self._mesh_timeseries_outputs(
            mesh_name, TIME_SERIES_OUTPUT_VARS_CELLS, truncate=False
        )
        return self._zmeta(ds)

    def zmeta_mesh_faces_timeseries_output(self, mesh_name: str) -> Dict:
        """Return kerchunk-style zarr reference metadata.

        Requires the 'zarr' and 'kerchunk' packages.

        Returns
        -------
        dict
            Dictionary of kerchunk-style zarr reference metadata.
        """
        ds = self._mesh_timeseries_outputs(
            mesh_name, TIME_SERIES_OUTPUT_VARS_FACES, truncate=False
        )
        return self._zmeta(ds)

    def zmeta_reference_lines_timeseries_output(self) -> Dict:
        """Return kerchunk-style zarr reference metadata.

        Requires the 'zarr' and 'kerchunk' packages.

        Returns
        -------
        dict
            Dictionary of kerchunk-style zarr reference metadata.
        """
        ds = self.reference_lines_timeseries_output()
        return self._zmeta(ds)

    def zmeta_reference_points_timeseries_output(self) -> Dict:
        """Return kerchunk-style zarr reference metadata.

        Requires the 'zarr' and 'kerchunk' packages.

        Returns
        -------
        dict
            Dictionary of kerchunk-style zarr reference metadata.
        """
        ds = self.reference_points_timeseries_output()
        return self._zmeta(ds)
