"""HEC-RAS Plan HDF class."""

from .geom import RasGeomHdf
from typing import Dict
from geopandas import GeoDataFrame


class RasPlanHdf(RasGeomHdf):
    """HEC-RAS Plan HDF class."""

    PLAN_INFO_PATH = "Plan Data/Plan Information"
    PLAN_PARAMS_PATH = "Plan Data/Plan Parameters"
    PRECIP_PATH = "Event Conditions/Meteorology/Precipitation"
    RESULTS_UNSTEADY_PATH = "Results/Unsteady"
    RESULTS_UNSTEADY_SUMMARY_PATH = f"{RESULTS_UNSTEADY_PATH}/Summary"
    VOLUME_ACCOUNTING_PATH = f"{RESULTS_UNSTEADY_PATH}/Volume Accounting"

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
