from .base import RasHdf
from .utils import convert_ras_hdf_string, get_first_hdf_group, hdf5_attrs_to_dict

import numpy as np
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely import (
    Polygon,
    Point,
    LineString,
    MultiLineString,
    MultiPolygon,
    polygonize,
)

from typing import List, Optional


class RasGeomHdf(RasHdf):
    GEOM_PATH = "Geometry"
    GEOM_STRUCTURES_PATH = f"{GEOM_PATH}/Structures"
    FLOW_AREA_2D_PATH = f"{GEOM_PATH}/2D Flow Areas"

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

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
        if isinstance(proj_wkt, bytes) or isinstance(proj_wkt, np.bytes_):
            proj_wkt = proj_wkt.decode("utf-8")
        return CRS.from_wkt(proj_wkt)

    def mesh_area_names(self) -> List[str]:
        """Return a list of the 2D mesh area names of
        the RAS geometry.

        Returns
        -------
        list
            A list of the 2D mesh area names (str) within the RAS geometry if 2D areas exist.
        """
        if "/Geometry/2D Flow Areas" not in self:
            return list()
        return list(
            [
                convert_ras_hdf_string(n)
                for n in self["/Geometry/2D Flow Areas/Attributes"][()]["Name"]
            ]
        )

    def mesh_areas(self) -> GeoDataFrame:
        """Return 2D flow area perimeter polygons.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D flow area perimeter polygons if 2D areas exist.
        """
        mesh_area_names = self.mesh_area_names()
        if not mesh_area_names:
            return GeoDataFrame()
        mesh_area_polygons = [
            Polygon(self[f"/Geometry/2D Flow Areas/{n}/Perimeter"][()])
            for n in mesh_area_names
        ]
        return GeoDataFrame(
            {"mesh_name": mesh_area_names, "geometry": mesh_area_polygons},
            geometry="geometry",
            crs=self.projection(),
        )

    def mesh_cell_polygons(self) -> GeoDataFrame:
        """Return the 2D flow mesh cell polygons.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D flow mesh cell polygons if 2D areas exist.
        """
        mesh_area_names = self.mesh_area_names()
        if not mesh_area_names:
            return GeoDataFrame()

        face_gdf = self.mesh_cell_faces()

        cell_dict = {"mesh_name": [], "cell_id": [], "geometry": []}
        for i, mesh_name in enumerate(mesh_area_names):
            cell_cnt = self["/Geometry/2D Flow Areas/Cell Info"][()][i][1]
            cell_ids = list(range(cell_cnt))
            cell_face_info = self[
                f"/Geometry/2D Flow Areas/{mesh_name}/Cells Face and Orientation Info"
            ][()]
            cell_face_values = self[
                f"/Geometry/2D Flow Areas/{mesh_name}/Cells Face and Orientation Values"
            ][()][:, 0]
            face_id_lists = list(
                np.vectorize(
                    lambda cell_id: str(
                        cell_face_values[
                            cell_face_info[cell_id][0] : cell_face_info[cell_id][0]
                            + cell_face_info[cell_id][1]
                        ]
                    )
                )(cell_ids)
            )
            mesh_faces = (
                face_gdf[face_gdf.mesh_name == mesh_name][["face_id", "geometry"]]
                .set_index("face_id")
                .to_numpy()
            )
            cell_dict["mesh_name"] += [mesh_name] * cell_cnt
            cell_dict["cell_id"] += cell_ids
            cell_dict["geometry"] += list(
                np.vectorize(
                    lambda face_id_list: polygonize(
                        np.ravel(
                            mesh_faces[
                                np.array(face_id_list.strip("[]").split()).astype(int)
                            ]
                        )
                    ).geoms[0]
                )(face_id_lists)
            )
        return GeoDataFrame(cell_dict, geometry="geometry", crs=self.projection())

    def mesh_cell_points(self) -> GeoDataFrame:
        """Return the 2D flow mesh cell points.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D flow mesh cell points if 2D areas exist.
        """
        mesh_area_names = self.mesh_area_names()
        if not mesh_area_names:
            return GeoDataFrame()
        pnt_dict = {"mesh_name": [], "cell_id": [], "geometry": []}
        for i, mesh_name in enumerate(mesh_area_names):
            starting_row, count = self["/Geometry/2D Flow Areas/Cell Info"][()][i]
            cell_pnt_coords = self["/Geometry/2D Flow Areas/Cell Points"][()][
                starting_row : starting_row + count
            ]
            pnt_dict["mesh_name"] += [mesh_name] * cell_pnt_coords.shape[0]
            pnt_dict["cell_id"] += range(count)
            pnt_dict["geometry"] += list(
                np.vectorize(lambda coords: Point(coords), signature="(n)->()")(
                    cell_pnt_coords
                )
            )
        return GeoDataFrame(pnt_dict, geometry="geometry", crs=self.projection())

    def mesh_cell_faces(self) -> GeoDataFrame:
        """Return the 2D flow mesh cell faces.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D flow mesh cell faces if 2D areas exist.
        """
        mesh_area_names = self.mesh_area_names()
        if not mesh_area_names:
            return GeoDataFrame()
        face_dict = {"mesh_name": [], "face_id": [], "geometry": []}
        for mesh_name in mesh_area_names:
            facepoints_index = self[
                f"/Geometry/2D Flow Areas/{mesh_name}/Faces FacePoint Indexes"
            ][()]
            facepoints_coordinates = self[
                f"/Geometry/2D Flow Areas/{mesh_name}/FacePoints Coordinate"
            ][()]
            faces_perimeter_info = self[
                f"/Geometry/2D Flow Areas/{mesh_name}/Faces Perimeter Info"
            ][()]
            faces_perimeter_values = self[
                f"/Geometry/2D Flow Areas/{mesh_name}/Faces Perimeter Values"
            ][()]
            face_id = -1
            for pnt_a_index, pnt_b_index in facepoints_index:
                face_id += 1
                face_dict["mesh_name"].append(mesh_name)
                face_dict["face_id"].append(face_id)
                coordinates = list()
                coordinates.append(facepoints_coordinates[pnt_a_index])
                starting_row, count = faces_perimeter_info[face_id]
                if count > 0:
                    coordinates += list(
                        faces_perimeter_values[starting_row : starting_row + count]
                    )
                coordinates.append(facepoints_coordinates[pnt_b_index])
                face_dict["geometry"].append(LineString(coordinates))
        return GeoDataFrame(face_dict, geometry="geometry", crs=self.projection())

    def get_geom_attrs(self):
        """Returns base geometry attributes from a HEC-RAS HDF file.

        Returns
        -------
        dict
            Dictionary filled with base geometry attributes.
        """
        return self.get_attrs(self.GEOM_PATH)

    def get_geom_structures_attrs(self):
        """Returns geometry structures attributes from a HEC-RAS HDF file.

        Returns
        -------
        dict
            Dictionary filled with geometry structures attributes.
        """
        return self.get_attrs(self.GEOM_STRUCTURES_PATH)

    def get_geom_2d_flow_area_attrs(self):
        """Returns geometry 2d flow area attributes from a HEC-RAS HDF file.

        Returns
        -------
        dict
            Dictionary filled with geometry 2d flow area attributes.
        """
        try:
            d2_flow_area = get_first_hdf_group(self.get(self.FLOW_AREA_2D_PATH))
        except AttributeError:
            raise AttributeError(
                f"Unable to get 2D Flow Area; {self.FLOW_AREA_2D_PATH} group not found in HDF5 file."
            )

        d2_flow_area_attrs = hdf5_attrs_to_dict(d2_flow_area.attrs)

        return d2_flow_area_attrs

    def bc_lines(self) -> GeoDataFrame:
        """Return the 2D mesh area boundary condition lines.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D mesh area boundary condition lines if they exist.
        """
        if "/Geometry/Boundary Condition Lines" not in self:
            return GeoDataFrame()
        bc_line_data = self["/Geometry/Boundary Condition Lines"]
        bc_line_ids = range(bc_line_data["Attributes"][()].shape[0])
        v_conv_str = np.vectorize(convert_ras_hdf_string)
        names = v_conv_str(bc_line_data["Attributes"][()]["Name"])
        mesh_names = v_conv_str(bc_line_data["Attributes"][()]["SA-2D"])
        types = v_conv_str(bc_line_data["Attributes"][()]["Type"])
        geoms = list()
        for pnt_start, pnt_cnt, part_start, part_cnt in bc_line_data["Polyline Info"][
            ()
        ]:
            points = bc_line_data["Polyline Points"][()][
                pnt_start : pnt_start + pnt_cnt
            ]
            if part_cnt == 1:
                geoms.append(LineString(points))
            else:
                parts = bc_line_data["Polyline Parts"][()][
                    part_start : part_start + part_cnt
                ]
                geoms.append(
                    MultiLineString(
                        list(
                            points[part_pnt_start : part_pnt_start + part_pnt_cnt]
                            for part_pnt_start, part_pnt_cnt in parts
                        )
                    )
                )
        return GeoDataFrame(
            {
                "bc_line_id": bc_line_ids,
                "name": names,
                "mesh_name": mesh_names,
                "type": types,
                "geometry": geoms,
            },
            geometry="geometry",
            crs=self.projection(),
        )

    def breaklines(self) -> GeoDataFrame:
        """Return the 2D mesh area breaklines.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D mesh area breaklines if they exist.
        """
        if "/Geometry/2D Flow Area Break Lines" not in self:
            return GeoDataFrame()
        bl_line_data = self["/Geometry/2D Flow Area Break Lines"]
        bl_line_ids = range(bl_line_data["Attributes"][()].shape[0])
        names = np.vectorize(convert_ras_hdf_string)(
            bl_line_data["Attributes"][()]["Name"]
        )
        geoms = list()
        for pnt_start, pnt_cnt, part_start, part_cnt in bl_line_data["Polyline Info"][
            ()
        ]:
            points = bl_line_data["Polyline Points"][()][
                pnt_start : pnt_start + pnt_cnt
            ]
            if part_cnt == 1:
                geoms.append(LineString(points))
            else:
                parts = bl_line_data["Polyline Parts"][()][
                    part_start : part_start + part_cnt
                ]
                geoms.append(
                    MultiLineString(
                        list(
                            points[part_pnt_start : part_pnt_start + part_pnt_cnt]
                            for part_pnt_start, part_pnt_cnt in parts
                        )
                    )
                )
        return GeoDataFrame(
            {"bl_id": bl_line_ids, "name": names, "geometry": geoms},
            geometry="geometry",
            crs=self.projection(),
        )

    def refinement_regions(self) -> GeoDataFrame:
        """Return the 2D mesh area refinement regions.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D mesh area refinement regions if they exist.
        """
        if "/Geometry/2D Flow Area Refinement Regions" not in self:
            return GeoDataFrame()
        rr_data = self["/Geometry/2D Flow Area Refinement Regions"]
        rr_ids = range(rr_data["Attributes"][()].shape[0])
        names = np.vectorize(convert_ras_hdf_string)(rr_data["Attributes"][()]["Name"])
        geoms = list()
        for pnt_start, pnt_cnt, part_start, part_cnt in rr_data["Polygon Info"][()]:
            points = rr_data["Polygon Points"][()][pnt_start : pnt_start + pnt_cnt]
            if part_cnt == 1:
                geoms.append(Polygon(points))
            else:
                parts = rr_data["Polygon Parts"][()][part_start : part_start + part_cnt]
                geoms.append(
                    MultiPolygon(
                        list(
                            points[part_pnt_start : part_pnt_start + part_pnt_cnt]
                            for part_pnt_start, part_pnt_cnt in parts
                        )
                    )
                )
        return GeoDataFrame(
            {"rr_id": rr_ids, "name": names, "geometry": geoms},
            geometry="geometry",
            crs=self.projection(),
        )

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
