from .geom import RasGeomHdf

from geopandas import GeoDataFrame


class RasPlanHdf(RasGeomHdf):

    def enroachment_points(self) -> GeoDataFrame:
        raise NotImplementedError
