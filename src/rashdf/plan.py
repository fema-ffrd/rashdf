from .geom import RasGeomHdf
from .utils import *
from typing import Dict
from geopandas import GeoDataFrame


class RasPlanHdf(RasGeomHdf):

    def get_plan_attrs(self) -> Dict:
        raise NotImplementedError

    def get_plan_info_attrs(self) -> Dict:
        raise NotImplementedError

    def get_plan_param_attrs(self) -> Dict:
        raise NotImplementedError

    def get_meteorology_precip_attrs(self) -> Dict:
        raise NotImplementedError

    def get_results_unsteady_attrs(self) -> Dict:
        raise NotImplementedError

    def get_results_summary_attrs(self) -> Dict:
        raise NotImplementedError

    def get_results_volume_accounting_attrs(self) -> Dict:
        raise NotImplementedError

    def enroachment_points(self) -> GeoDataFrame:
        raise NotImplementedError
