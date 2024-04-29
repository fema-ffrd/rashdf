from .geom import RasGeomHdf
from .base import RasHdf
from typing import Dict
from geopandas import GeoDataFrame


class RasPlanHdf(RasHdf):

    def __init__(self, name: str):
        super().__init__(name)
        self.plan_info_path = "Plan Data/Plan Information"
        self.plan_params_path = "Plan Data/Plan Parameters"
        self.meteorology_precip_path = "Event Conditions/Meteorology/Precipitation"
        self.results_unsteady_path = "Results/Unsteady"
        self.results_summary_path = "Results/Unsteady/Summary"
        self.volume_accounting_path = "Results/Unsteady/Summary/Volume Accounting"

    def get_plan_info_attrs(self) -> Dict:
        """Returns plan information attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary filled with plan information attributes.
        """
        return self.get_attrs(self.plan_info_path)

    def get_plan_param_attrs(self) -> Dict:
        """Returns plan parameter attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary filled with plan parameter attributes.
        """
        return self.get_attrs(self.plan_params_path)

    def get_meteorology_precip_attrs(self) -> Dict:
        """Returns precipitation attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary filled with precipitation attributes.
        """
        return self.get_attrs(self.meteorology_precip_path)

    def get_results_unsteady_attrs(self) -> Dict:
        """Returns unsteady attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary filled with unsteady attributes.
        """
        return self.get_attrs(self.results_unsteady_path)

    def get_results_summary_attrs(self) -> Dict:
        """Returns results summary attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary filled with results summary attributes.
        """
        return self.get_attrs(self.results_summary_path)

    def get_results_volume_accounting_attrs(self) -> Dict:
        """Returns volume accounting attributes from a HEC-RAS HDF plan file.

        Returns
        -------
        dict
            Dictionary filled with volume accounting attributes.
        """
        return self.get_attrs(self.volume_accounting_path)

    def enroachment_points(self) -> GeoDataFrame:
        raise NotImplementedError
