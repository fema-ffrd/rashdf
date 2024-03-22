from .base import RasHdf
from .utils import convert_ras_hdf_string

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely import Polygon, Point, LineString

from typing import Optional


class RasGeomHdf(RasHdf):

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

    def mesh_areas(self) -> GeoDataFrame:
        """Return 2D flow area perimeter polygons.
        
        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D flow area perimeter polygons.
        """
        mesh_area_names = [convert_ras_hdf_string(n) for n in self["/Geometry/2D Flow Areas/Attributes"]["Name"][()]]
        mesh_area_polygons = [Polygon(self[f"/Geometry/2D Flow Areas/{n}/Perimeter"]) for n in mesh_area_names]
        return GeoDataFrame({"name" : mesh_area_names, "geometry" : mesh_area_polygons}, geometry="geometry")

    def mesh_cell_polygons(self) -> GeoDataFrame:
        """Return the 2D flow mesh cell polygons.
        
        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D flow mesh cell polygons.
        """
        raise NotImplementedError

    def mesh_cell_points(self) -> GeoDataFrame:
        """Return the 2D flow mesh cell points.
        
        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D flow mesh cell points.
        """
        mesh_area_names = [convert_ras_hdf_string(n) for n in self["/Geometry/2D Flow Areas/Attributes"]["Name"][()]]
        pnt_dict = {"mesh_name":[], "cell_id":[], "geometry":[]}
        for mesh_name in mesh_area_names:
            cell_pnt_coords = self[f"/Geometry/2D Flow Areas/{mesh_name}/Cells Center Coordinate"][()]
            cell_cnt = len(cell_pnt_coords)
            pnt_dict["mesh_name"] += [mesh_name]*cell_cnt
            pnt_dict["cell_id"] += range(cell_cnt)
            pnt_dict["geometry"] += [Point(*coords) for coords in cell_pnt_coords]
        return GeoDataFrame(pnt_dict, geometry="geometry")

    def mesh_cell_faces(self) -> GeoDataFrame:
        """Return the 2D flow mesh cell faces.
        
        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D flow mesh cell faces.
        """
        mesh_area_names = [convert_ras_hdf_string(n) for n in self["/Geometry/2D Flow Areas/Attributes"]["Name"][()]]
        face_dict = {"mesh_name":[], "face_id":[], "geometry":[]}
        for mesh_name in mesh_area_names:
            facepoints_index = self[f"/Geometry/2D Flow Areas/{mesh_name}/Faces FacePoint Indexes"][()]
            facepoints_coordinates = self[f"/Geometry/2D Flow Areas/{mesh_name}/FacePoints Coordinate"][()]
            faces_perimeter_info = self[f"/Geometry/2D Flow Areas/{mesh_name}/Faces Perimeter Info"][()]
            faces_perimeter_values = self[f"/Geometry/2D Flow Areas/{mesh_name}/Faces Perimeter Values"][()]
            face_id = -1
            for pnt_a_index, pnt_b_index in facepoints_index:
                face_id+=1
                face_dict["mesh_name"].append(mesh_name)
                face_dict["face_id"].append(face_id)
                coordinates = list()
                coordinates.append(facepoints_coordinates[pnt_a_index])
                starting_row, count = faces_perimeter_info[face_id]
                if count > 0:
                    coordinates += list(faces_perimeter_values[starting_row:starting_row+count])
                coordinates.append(facepoints_coordinates[pnt_b_index])
                face_dict["geometry"].append(LineString(coordinates))
        return GeoDataFrame(face_dict, geometry="geometry")

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
