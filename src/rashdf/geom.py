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
    
    def mesh_area_names(self) -> Optional[list]:
        """Return a list of the 2D mesh area names of 
        the RAS geometry.
        
        Returns
        -------
        list
            A list of the 2D mesh area names (str) within the RAS geometry if 2D areas exist.
            Otherwise, returns None.
        """
        if "/Geometry/2D Flow Areas" not in self:
            return None
        return list([convert_ras_hdf_string(n) for n in self["/Geometry/2D Flow Areas/Attributes"]["Name"][()]])

    def mesh_areas(self) -> Optional[GeoDataFrame]:
        """Return 2D flow area perimeter polygons.
        
        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D flow area perimeter polygons if 2D areas exist.
            Otherwise, returns None.
        """
        # get mesh area names
        mesh_area_names = self.mesh_area_names()
        if not mesh_area_names:
            return None
        mesh_area_polygons = [Polygon(self[f"/Geometry/2D Flow Areas/{n}/Perimeter"]) for n in mesh_area_names]
        return GeoDataFrame({"mesh_name" : mesh_area_names, "geometry" : mesh_area_polygons}, geometry="geometry", crs=self.projection())

    def mesh_cell_polygons(self) -> Optional[GeoDataFrame]:
        """Return the 2D flow mesh cell polygons.
        
        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D flow mesh cell polygons if 2D areas exist.
            Otherwise, returns None.
        """
        # get mesh area names
        mesh_area_names = self.mesh_area_names()
        if not mesh_area_names:
            return None
        
        # recursive coord cleaning
        def clean_coords(end_coord: list = None) -> None:
            if not end_coord:
                target = raw_coords.pop(0)
            else:
                i = -1
                for face in raw_coords:
                    i+=1
                    if end_coord == face[0]:
                        target = raw_coords.pop(i)
                    elif end_coord == face[-1]:
                        target = raw_coords.pop(i)
                        target=list(reversed(target))
            [cleaned_coords.append(coord) for coord in target]
            if raw_coords: clean_coords(target[-1])
        def polygonize(face_coord_list) -> Polygon:
            global cleaned_coords; global raw_coords
            cleaned_coords=list(); raw_coords=face_coord_list
            clean_coords()
            return Polygon(cleaned_coords)

        # assemble face dict
        face_df = self.mesh_cell_faces()
        face_df["face_code"] = face_df["face_id"].astype(str)+face_df["mesh_name"]
        face_df["coords"] = face_df["geometry"].apply(lambda face: face.coords)
        face_dict = face_df[["face_code", "coords"]].set_index("face_code", inplace=False).to_dict()["coords"]

        # assemble cell poly df
        cell_dict = {"mesh_name":[], "cell_id":[], "face_id_list":[]}
        for mesh_name in mesh_area_names:
            cell_face_info = self[f"/Geometry/2D Flow Areas/{mesh_name}/Cells Face and Orientation Info"][()]
            cell_face_values = self[f"/Geometry/2D Flow Areas/{mesh_name}/Cells Face and Orientation Values"][()]
            cell_cnt = cell_face_info.shape[0]
            cell_dict["mesh_name"] += [mesh_name]*cell_cnt
            cell_dict["cell_id"] += range(cell_cnt)
            cell_dict["face_id_list"] += list(cell_face_values[starting_row:starting_row+count, 0] for starting_row, count in cell_face_info)
        cell_dict["face_cnt"] = list([len(l) for l in cell_dict["face_id_list"]])
        cell_df = pd.DataFrame(cell_dict)
        cell_df = cell_df[cell_df.face_cnt > 1]
        cell_df["face_coord_list"] = cell_df.apply(lambda x: list([face_dict[str(face_id)+x.mesh_name] for face_id in x.face_id_list]), axis=1)
        cell_df["geometry"] = cell_df["face_coord_list"].apply(polygonize)

        return GeoDataFrame(cell_df[["mesh_name", "cell_id", "geometry"]], geometry="geometry", crs=self.projection())

    def mesh_cell_points(self) -> Optional[GeoDataFrame]:
        """Return the 2D flow mesh cell points.
        
        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D flow mesh cell points if 2D areas exist.
            Otherwise, returns None.
        """
        mesh_area_names = self.mesh_area_names()
        if not mesh_area_names:
            return None
        pnt_dict = {"mesh_name":[], "cell_id":[], "geometry":[]}
        for mesh_name in mesh_area_names:
            cell_pnt_coords = self[f"/Geometry/2D Flow Areas/{mesh_name}/Cells Center Coordinate"][()]
            cell_cnt = cell_pnt_coords.shape[0]
            pnt_dict["mesh_name"] += [mesh_name]*cell_cnt
            pnt_dict["cell_id"] += range(cell_cnt)
            pnt_dict["geometry"] += [Point(*coords) for coords in cell_pnt_coords]
        return GeoDataFrame(pnt_dict, geometry="geometry", crs=self.projection())

    def mesh_cell_faces(self) -> Optional[GeoDataFrame]:
        """Return the 2D flow mesh cell faces.
        
        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D flow mesh cell faces if 2D areas exist.
            Otherwise, returns None.
        """
        mesh_area_names = self.mesh_area_names()
        if not mesh_area_names:
            return None        
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
        return GeoDataFrame(face_dict, geometry="geometry", crs=self.projection())

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
