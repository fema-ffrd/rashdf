from .base import RasHdf

import numpy as np
from geopandas import GeoDataFrame
from pyproj import CRS

from typing import Optional


class RasGeomHdf(RasHdf):

    def __init__(self, hdf_file: str):
        """Open a HEC-RAS Geometry HDF file."""
        super().__init__(hdf_file)

    def projection(self) -> Optional[CRS]:
        """Return the projection of the RAS geometry as a
        pyproj.CRS object.
        
        Returns
        -------
        CRS
            The projection of the RAS geometry.
        """
        proj_wkt = self.attrs.get("Projection")
        if proj_wkt is None:
            return None
        if type(proj_wkt) == bytes or type(proj_wkt) == np.bytes_:
            proj_wkt = proj_wkt.decode("utf-8")
        return CRS.from_wkt(proj_wkt)

    def d2_flow_areas(self) -> GeoDataFrame:
        """Return 2D flow area perimeter polygons.
        
        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D flow area perimeter polygons.
        """
        raise NotImplementedError

    def mesh_cell_polygons(self) -> GeoDataFrame:
        """Return the 2D flow mesh cell polygons.
        
        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D flow mesh cell polygons.
        """
        raise NotImplementedError

    def mesh_cell_points(self) -> GeoDataFrame:
        raise NotImplementedError

    def mesh_cell_faces(self) -> GeoDataFrame:
        raise NotImplementedError

    def bc_lines(self) -> GeoDataFrame:
        raise NotImplementedError

    def breaklines(self) -> GeoDataFrame:
        raise NotImplementedError

    def refinement_regions(self) -> GeoDataFrame:
        raise NotImplementedError

    def connections(self) -> GeoDataFrame:
        raise NotImplementedError

    def ic_points(self) -> GeoDataFrame:
        raise NotImplementedError

    def reference_lines(self) -> GeoDataFrame:
        raise NotImplementedError

    def reference_points(self) -> GeoDataFrame:
        raise NotImplementedError

    def structures(self) -> GeoDataFrame:
        raise NotImplementedError

    def pump_stations(self) -> GeoDataFrame:
        raise NotImplementedError

    def mannings_calibration_regions(self) -> GeoDataFrame:
        raise NotImplementedError

    def classification_polygons(self) -> GeoDataFrame:
        raise NotImplementedError

    def terrain_modifications(self) -> GeoDataFrame:
        raise NotImplementedError

    def cross_sections(self) -> GeoDataFrame:
        raise NotImplementedError

    def river_reaches(self) -> GeoDataFrame:
        raise NotImplementedError

    def flowpaths(self) -> GeoDataFrame:
        raise NotImplementedError

    def bank_points(self) -> GeoDataFrame:
        raise NotImplementedError
    
    def bank_lines(self) -> GeoDataFrame:
        raise NotImplementedError

    def ineffective_areas(self) -> GeoDataFrame:
        raise NotImplementedError

    def ineffective_points(self) -> GeoDataFrame:
        raise NotImplementedError

    def blocked_obstructions(self) -> GeoDataFrame:
        raise NotImplementedError
