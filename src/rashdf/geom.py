from .base import RasHdf

import geopandas as gpd


class RasGeomHdf(RasHdf):

    def __init__(self, hdf_file: str):
        """Open a HEC-RAS Geometry HDF file."""
        super().__init__(hdf_file)

    def d2_flow_areas(self) -> gpd.GeoDataFrame:
        pass

    def mesh_cell_polygons(self) -> gpd.GeoDataFrame:
        pass

    def mesh_cell_points(self) -> gpd.GeoDataFrame:
        pass

    def mesh_cell_faces(self) -> gpd.GeoDataFrame:
        pass

    def bc_lines(self) -> gpd.GeoDataFrame:
        pass

    def breaklines(self) -> gpd.GeoDataFrame:
        pass

    def refinement_regions(self) -> gpd.GeoDataFrame:
        pass

    def connections(self) -> gpd.GeoDataFrame:
        pass

    def ic_points(self) -> gpd.GeoDataFrame:
        pass

    def reference_lines(self) -> gpd.GeoDataFrame:
        pass

    def reference_points(self) -> gpd.GeoDataFrame:
        pass

    def structures(self) -> gpd.GeoDataFrame:
        pass

    def pump_stations(self) -> gpd.GeoDataFrame:
        pass

    def mannings_calibration_regions(self) -> gpd.GeoDataFrame:
        pass

    def classification_polygons(self) -> gpd.GeoDataFrame:
        pass

    def terrain_modifications(self) -> gpd.GeoDataFrame:
        pass

    def cross_sections(self) -> gpd.GeoDataFrame:
        pass

    def river_reaches(self) -> gpd.GeoDataFrame:
        pass

    def flowpaths(self) -> gpd.GeoDataFrame:
        pass

    def bank_points(self) -> gpd.GeoDataFrame:
        pass
    
    def bank_lines(self) -> gpd.GeoDataFrame:
        pass

    def ineffective_areas(self) -> gpd.GeoDataFrame:
        pass

    def ineffective_points(self) -> gpd.GeoDataFrame:
        pass

    def blocked_obstructions(self) -> gpd.GeoDataFrame:
        pass
