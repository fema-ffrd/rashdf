from .geom import RasGeomHdf
from typing import Dict
from geopandas import GeoDataFrame


class RasPlanHdf(RasGeomHdf):
    PLAN_INFO_PATH = "Plan Data/Plan Information"
    PLAN_PARAMS_PATH = "Plan Data/Plan Parameters"
    PRECIP_PATH = "Event Conditions/Meteorology/Precipitation"
    RESULTS_UNSTEADY_PATH = "Results/Unsteady"
    RESULTS_UNSTEADY_SUMMARY_PATH = f"{RESULTS_UNSTEADY_PATH}/Summary"
    VOLUME_ACCOUNTING_PATH = f"{RESULTS_UNSTEADY_PATH}/Volume Accounting"

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

    def get_plan_info_attrs(self) -> Dict:
        """Returns plan information attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary filled with plan information attributes.
        """
        return self.get_attrs(self.PLAN_INFO_PATH)

    def get_plan_param_attrs(self) -> Dict:
        """Returns plan parameter attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary filled with plan parameter attributes.
        """
        return self.get_attrs(self.PLAN_PARAMS_PATH)

    def get_meteorology_precip_attrs(self) -> Dict:
        """Returns precipitation attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary filled with precipitation attributes.
        """
        return self.get_attrs(self.PRECIP_PATH)

    def get_results_unsteady_attrs(self) -> Dict:
        """Returns unsteady attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary filled with unsteady attributes.
        """
        return self.get_attrs(self.RESULTS_UNSTEADY_PATH)

    def get_results_unsteady_summary_attrs(self) -> Dict:
        """Returns results unsteady summary attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary filled with results summary attributes.
        """
        return self.get_attrs(self.RESULTS_UNSTEADY_SUMMARY_PATH)

    def get_results_volume_accounting_attrs(self) -> Dict:
        """Returns volume accounting attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary filled with volume accounting attributes.
        """
        return self.get_attrs(self.VOLUME_ACCOUNTING_PATH)

    def enroachment_points(self) -> GeoDataFrame:
        raise NotImplementedError
