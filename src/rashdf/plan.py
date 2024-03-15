from .geom import RasGeomHdf

import geopandas as gpd


class RasPlanHdf(RasGeomHdf):

    def __init__(self, hdf_file: str):
        """Open a HEC-RAS Plan HDF file."""
        super().__init__(hdf_file)

    def enroachment_points(self) -> gpd.GeoDataFrame:
        pass
