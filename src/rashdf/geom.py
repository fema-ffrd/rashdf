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
        mesh_area_names = [convert_ras_hdf_string(n) for n in self["/Geometry/2D Flow Areas/Attributes"]["Name"][()]]
        cell_df_list = list()
        for mesh_name in mesh_area_names:
            cell_face_info = self[f"/Geometry/2D Flow Areas/{mesh_name}/Cells Face and Orientation Info"][()]
            cell_face_values = self[f"/Geometry/2D Flow Areas/{mesh_name}/Cells Face and Orientation Values"][()]
            cell_id = -1
            cell_dict = {"mesh_name":[], "cell_id":[], "face_id_list":[], "face_cnt":[]}
            for starting_row, count in cell_face_info:
                cell_id+=1
                cell_dict["mesh_name"].append(mesh_name)
                cell_dict["cell_id"].append(cell_id)
                face_id_list = list(cell_face_values[starting_row:starting_row+count, 0])
                cell_dict["face_id_list"].append(face_id_list)
                cell_dict["face_cnt"].append(len(face_id_list))
            cell_df = pd.DataFrame(cell_dict)
            cell_df = cell_df[cell_df["face_cnt"] > 1]
            face_dict = self.mesh_cell_faces()[["face_id", "geometry"]].set_index("face_id", inplace=False).to_dict()["geometry"]
            def polygonize(face_id_list):
                ring_coords = list()
                for fid in face_id_list:
                    ring_coords += face_dict[fid].coords
                return Polygon(set(ring_coords))
            cell_df["geometry"] = cell_df["face_id_list"].apply(lambda x: polygonize(x))
            cell_df_list.append(cell_df)
        return GeoDataFrame(pd.concat(cell_df_list).drop("face_id_list", axis=1, inplace=False), geometry="geometry")

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
