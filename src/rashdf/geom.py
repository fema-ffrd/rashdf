"""HEC-RAS Geometry HDF class."""

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely import (
    Geometry,
    Polygon,
    Point,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    polygonize_full,
)

from typing import Dict, List, Optional, Union


from .base import RasHdf
from .utils import (
    convert_ras_hdf_string,
    convert_ras_hdf_value,
    get_first_hdf_group,
    hdf5_attrs_to_dict,
)


class RasGeomHdfError(Exception):
    """HEC-RAS Plan HDF error class."""

    pass


class RasGeomHdf(RasHdf):
    """HEC-RAS Geometry HDF class."""

    GEOM_PATH = "Geometry"
    GEOM_STRUCTURES_PATH = f"{GEOM_PATH}/Structures"
    FLOW_AREA_2D_PATH = f"{GEOM_PATH}/2D Flow Areas"
    BC_LINES_PATH = f"{GEOM_PATH}/Boundary Condition Lines"
    BREAKLINES_PATH = f"{GEOM_PATH}/2D Flow Area Break Lines"
    REFERENCE_LINES_PATH = f"{GEOM_PATH}/Reference Lines"
    REFERENCE_POINTS_PATH = f"{GEOM_PATH}/Reference Points"
    CROSS_SECTIONS = f"{GEOM_PATH}/Cross Sections"
    RIVER_CENTERLINES = f"{GEOM_PATH}/River Centerlines"

    def __init__(self, name: str, **kwargs):
        """Open a HEC-RAS Geometry HDF file.

        Parameters
        ----------
        name : str
            The path to the RAS Geometry HDF file.
        kwargs : dict
            Additional keyword arguments to pass to h5py.File
        """
        super().__init__(name, **kwargs)

    def projection(self) -> Optional[CRS]:
        """Return the projection of the RAS geometry as a pyproj.CRS object.

        Returns
        -------
        pyproj.CRS or None
            The projection of the RAS geometry.
        """
        proj_wkt = self.attrs.get("Projection")
        if proj_wkt is None:
            return None
        if isinstance(proj_wkt, bytes) or isinstance(proj_wkt, np.bytes_):
            proj_wkt = proj_wkt.decode("utf-8")
        return CRS.from_wkt(proj_wkt)

    def mesh_area_names(self) -> List[str]:
        """Return a list of the 2D mesh area names of the RAS geometry.

        Returns
        -------
        List[str]
            A list of the 2D mesh area names (str) within the RAS geometry if 2D areas exist.
        """
        if self.FLOW_AREA_2D_PATH not in self:
            return list()
        return list(
            [
                convert_ras_hdf_string(n)
                for n in self[f"{self.FLOW_AREA_2D_PATH}/Attributes"][()]["Name"]
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
            Polygon(self[f"{self.FLOW_AREA_2D_PATH}/{n}/Perimeter"][()])
            for n in mesh_area_names
        ]
        return GeoDataFrame(
            {"mesh_name": mesh_area_names, "geometry": mesh_area_polygons},
            geometry="geometry",
            crs=self.projection(),
        )

    def mesh_cell_polygons(self) -> GeoDataFrame:
        """Return 2D flow mesh cell polygons.

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
            cell_cnt = self[f"{self.FLOW_AREA_2D_PATH}/Cell Info"][()][i][1]
            cell_ids = list(range(cell_cnt))
            cell_face_info = self[
                f"{self.FLOW_AREA_2D_PATH}/{mesh_name}/Cells Face and Orientation Info"
            ][()]
            cell_face_values = self[
                f"{self.FLOW_AREA_2D_PATH}/{mesh_name}/Cells Face and Orientation Values"
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
                    lambda face_id_list: (
                        lambda geom_col: Polygon((geom_col[0] or geom_col[3]).geoms[0])
                    )(
                        polygonize_full(
                            np.ravel(
                                mesh_faces[
                                    np.array(face_id_list.strip("[]").split()).astype(
                                        int
                                    )
                                ]
                            )
                        )
                    )
                )(face_id_lists)
            )
        return GeoDataFrame(cell_dict, geometry="geometry", crs=self.projection())

    def mesh_cell_points(self) -> GeoDataFrame:
        """Return 2D flow mesh cell points.

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
            starting_row, count = self[f"{self.FLOW_AREA_2D_PATH}/Cell Info"][()][i]
            cell_pnt_coords = self[f"{self.FLOW_AREA_2D_PATH}/Cell Points"][()][
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
        """Return 2D flow mesh cell faces.

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
                f"{self.FLOW_AREA_2D_PATH}/{mesh_name}/Faces FacePoint Indexes"
            ][()]
            facepoints_coordinates = self[
                f"{self.FLOW_AREA_2D_PATH}/{mesh_name}/FacePoints Coordinate"
            ][()]
            faces_perimeter_info = self[
                f"{self.FLOW_AREA_2D_PATH}/{mesh_name}/Faces Perimeter Info"
            ][()]
            faces_perimeter_values = self[
                f"{self.FLOW_AREA_2D_PATH}/{mesh_name}/Faces Perimeter Values"
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

    def get_geom_attrs(self) -> Dict:
        """Return base geometry attributes from a HEC-RAS HDF file.

        Returns
        -------
        dict
            Dictionary filled with base geometry attributes.
        """
        return self.get_attrs(self.GEOM_PATH)

    def get_geom_structures_attrs(self) -> Dict:
        """Return geometry structures attributes from a HEC-RAS HDF file.

        Returns
        -------
        dict
            Dictionary filled with geometry structures attributes.
        """
        return self.get_attrs(self.GEOM_STRUCTURES_PATH)

    def get_geom_2d_flow_area_attrs(self) -> Dict:
        """Return geometry 2d flow area attributes from a HEC-RAS HDF file.

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

    def _get_polylines(
        self,
        path: str,
        info_name: str = "Polyline Info",
        parts_name: str = "Polyline Parts",
        points_name: str = "Polyline Points",
    ) -> List[Geometry]:
        polyline_info_path = f"{path}/{info_name}"
        polyline_parts_path = f"{path}/{parts_name}"
        polyline_points_path = f"{path}/{points_name}"

        polyline_info = self[polyline_info_path][()]
        polyline_parts = self[polyline_parts_path][()]
        polyline_points = self[polyline_points_path][()]

        geoms = []
        for pnt_start, pnt_cnt, part_start, part_cnt in polyline_info:
            points = polyline_points[pnt_start : pnt_start + pnt_cnt]
            if part_cnt == 1:
                geoms.append(LineString(points))
            else:
                parts = polyline_parts[part_start : part_start + part_cnt]
                geoms.append(
                    MultiLineString(
                        list(
                            points[part_pnt_start : part_pnt_start + part_pnt_cnt]
                            for part_pnt_start, part_pnt_cnt in parts
                        )
                    )
                )
        return geoms

    def bc_lines(self) -> GeoDataFrame:
        """Return 2D mesh area boundary condition lines.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D mesh area boundary condition lines if they exist.
        """
        if self.BC_LINES_PATH not in self:
            return GeoDataFrame()
        bc_line_data = self[self.BC_LINES_PATH]
        bc_line_ids = range(bc_line_data["Attributes"][()].shape[0])
        v_conv_str = np.vectorize(convert_ras_hdf_string)
        names = v_conv_str(bc_line_data["Attributes"][()]["Name"])
        mesh_names = v_conv_str(bc_line_data["Attributes"][()]["SA-2D"])
        types = v_conv_str(bc_line_data["Attributes"][()]["Type"])
        geoms = self._get_polylines(self.BC_LINES_PATH)
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
        """Return 2D mesh area breaklines.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the 2D mesh area breaklines if they exist.
        """
        if self.BREAKLINES_PATH not in self:
            return GeoDataFrame()
        bl_line_data = self[self.BREAKLINES_PATH]
        bl_line_ids = range(bl_line_data["Attributes"][()].shape[0])
        names = np.vectorize(convert_ras_hdf_string)(
            bl_line_data["Attributes"][()]["Name"]
        )
        geoms = self._get_polylines(self.BREAKLINES_PATH)
        return GeoDataFrame(
            {"bl_id": bl_line_ids, "name": names, "geometry": geoms},
            geometry="geometry",
            crs=self.projection(),
        )

    def refinement_regions(self) -> GeoDataFrame:
        """Return 2D mesh area refinement regions.

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

    def structures(self, datetime_to_str: bool = False) -> GeoDataFrame:
        """Return the model structures.

        Parameters
        ----------
        datetime_to_str : bool, optional
            If True, convert datetime values to string format (default: False).

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the model structures if they exist.
        """
        if self.GEOM_STRUCTURES_PATH not in self:
            return GeoDataFrame()
        struct_data = self[self.GEOM_STRUCTURES_PATH]
        v_conv_val = np.vectorize(convert_ras_hdf_value)
        sd_attrs = struct_data["Attributes"][()]
        struct_dict = {"struct_id": range(sd_attrs.shape[0])}
        struct_dict.update(
            {name: v_conv_val(sd_attrs[name]) for name in sd_attrs.dtype.names}
        )
        geoms = self._get_polylines(
            self.GEOM_STRUCTURES_PATH,
            info_name="Centerline Info",
            parts_name="Centerline Parts",
            points_name="Centerline Points",
        )
        struct_gdf = GeoDataFrame(
            struct_dict,
            geometry=geoms,
            crs=self.projection(),
        )
        if datetime_to_str:
            struct_gdf["Last Edited"] = struct_gdf["Last Edited"].apply(
                lambda x: pd.Timestamp.isoformat(x)
            )
        return struct_gdf

    def connections(self) -> GeoDataFrame:  # noqa D102
        raise NotImplementedError

    def ic_points(self) -> GeoDataFrame:  # noqa D102
        raise NotImplementedError

    def _reference_lines_points_names(
        self, reftype: str = "lines", mesh_name: Optional[str] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Return reference line names.

        If a mesh name is provided, return a list of the reference line names for that mesh area.
        If no mesh name is provided, return a dictionary of mesh names and their reference line names.

        Parameters
        ----------
        mesh_name : str, optional
            The name of the mesh area for which to return reference line names.

        Returns
        -------
        Union[Dict[str, List[str]], List[str]]
            A dictionary of mesh names and their reference line names if mesh_name is None.
            A list of reference line names for the specified mesh area if mesh_name is not None.
        """
        if reftype == "lines":
            path = self.REFERENCE_LINES_PATH
            sa_2d_field = "SA-2D"
        elif reftype == "points":
            path = self.REFERENCE_POINTS_PATH
            sa_2d_field = "SA/2D"
        else:
            raise RasGeomHdfError(
                f"Invalid reference type: {reftype} -- must be 'lines' or 'points'."
            )
        attributes_path = f"{path}/Attributes"
        if mesh_name is None and attributes_path not in self:
            return {m: [] for m in self.mesh_area_names()}
        if mesh_name is not None and attributes_path not in self:
            return []
        attributes = self[attributes_path][()]
        v_conv_str = np.vectorize(convert_ras_hdf_string)
        names = np.vectorize(convert_ras_hdf_string)(attributes["Name"])
        if mesh_name is not None:
            return names[v_conv_str(attributes[sa_2d_field]) == mesh_name].tolist()
        mesh_names = np.vectorize(convert_ras_hdf_string)(attributes[sa_2d_field])
        return {m: names[mesh_names == m].tolist() for m in np.unique(mesh_names)}

    def reference_lines_names(
        self, mesh_name: Optional[str] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Return reference line names.

        If a mesh name is provided, return a list of the reference line names for that mesh area.
        If no mesh name is provided, return a dictionary of mesh names and their reference line names.

        Parameters
        ----------
        mesh_name : str, optional
            The name of the mesh area for which to return reference line names.

        Returns
        -------
        Union[Dict[str, List[str]], List[str]]
            A dictionary of mesh names and their reference line names if mesh_name is None.
            A list of reference line names for the specified mesh area if mesh_name is not None.
        """
        return self._reference_lines_points_names("lines", mesh_name)

    def reference_points_names(
        self, mesh_name: Optional[str] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Return reference point names.

        If a mesh name is provided, return a list of the reference point names for that mesh area.
        If no mesh name is provided, return a dictionary of mesh names and their reference point names.

        Parameters
        ----------
        mesh_name : str, optional
            The name of the mesh area for which to return reference point names.

        Returns
        -------
        Union[Dict[str, List[str]], List[str]]
            A dictionary of mesh names and their reference point names if mesh_name is None.
            A list of reference point names for the specified mesh area if mesh_name is not None.
        """
        return self._reference_lines_points_names("points", mesh_name)

    def reference_lines(self) -> GeoDataFrame:
        """Return the reference lines geometry and attributes.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the reference lines if they exist.
        """
        attributes_path = f"{self.REFERENCE_LINES_PATH}/Attributes"
        if attributes_path not in self:
            return GeoDataFrame()
        attributes = self[attributes_path][()]
        refline_ids = range(attributes.shape[0])
        v_conv_str = np.vectorize(convert_ras_hdf_string)
        names = v_conv_str(attributes["Name"])
        mesh_names = v_conv_str(attributes["SA-2D"])
        try:
            types = v_conv_str(attributes["Type"])
        except ValueError:
            # "Type" field doesn't exist -- observed in some RAS HDF files
            types = np.array([""] * attributes.shape[0])
        geoms = self._get_polylines(self.REFERENCE_LINES_PATH)
        return GeoDataFrame(
            {
                "refln_id": refline_ids,
                "refln_name": names,
                "mesh_name": mesh_names,
                "type": types,
                "geometry": geoms,
            },
            geometry="geometry",
            crs=self.projection(),
        )

    def reference_points(self) -> GeoDataFrame:
        """Return the reference points geometry and attributes.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the reference points if they exist.
        """
        attributes_path = f"{self.REFERENCE_POINTS_PATH}/Attributes"
        if attributes_path not in self:
            return GeoDataFrame()
        ref_points_group = self[self.REFERENCE_POINTS_PATH]
        attributes = ref_points_group["Attributes"][:]
        v_conv_str = np.vectorize(convert_ras_hdf_string)
        names = v_conv_str(attributes["Name"])
        mesh_names = v_conv_str(attributes["SA/2D"])
        cell_id = attributes["Cell Index"]
        points = ref_points_group["Points"][()]
        return GeoDataFrame(
            {
                "refpt_id": range(attributes.shape[0]),
                "refpt_name": names,
                "mesh_name": mesh_names,
                "cell_id": cell_id,
                "geometry": list(map(Point, points)),
            },
            geometry="geometry",
            crs=self.projection(),
        )

    def pump_stations(self) -> GeoDataFrame:  # noqa D102
        raise NotImplementedError

    def mannings_calibration_regions(self) -> GeoDataFrame:  # noqa D102
        raise NotImplementedError

    def classification_polygons(self) -> GeoDataFrame:  # noqa D102
        raise NotImplementedError

    def terrain_modifications(self) -> GeoDataFrame:  # noqa D102
        raise NotImplementedError

    def cross_sections(self, datetime_to_str: bool = False) -> GeoDataFrame:
        """Return the model 1D cross sections.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the model 1D cross sections if they exist.
        """
        if self.CROSS_SECTIONS not in self:
            return GeoDataFrame()

        xs_data = self[self.CROSS_SECTIONS]
        v_conv_val = np.vectorize(convert_ras_hdf_value)
        xs_attrs = xs_data["Attributes"][()]
        xs_dict = {"xs_id": range(xs_attrs.shape[0])}
        xs_dict.update(
            {name: v_conv_val(xs_attrs[name]) for name in xs_attrs.dtype.names}
        )
        geoms = self._get_polylines(self.CROSS_SECTIONS)
        xs_gdf = GeoDataFrame(
            xs_dict,
            geometry=geoms,
            crs=self.projection(),
        )
        if datetime_to_str:
            xs_gdf["Last Edited"] = xs_gdf["Last Edited"].apply(
                lambda x: pd.Timestamp.isoformat(x)
            )
        return xs_gdf

    def river_reaches(self, datetime_to_str: bool = False) -> GeoDataFrame:
        """Return the model 1D river reach lines.

        Returns
        -------
        GeoDataFrame
            A GeoDataFrame containing the model 1D river reach lines if they exist.
        """
        if self.RIVER_CENTERLINES not in self:
            return GeoDataFrame()

        river_data = self[self.RIVER_CENTERLINES]
        v_conv_val = np.vectorize(convert_ras_hdf_value)
        river_attrs = river_data["Attributes"][()]
        river_dict = {"river_id": range(river_attrs.shape[0])}
        river_dict.update(
            {name: v_conv_val(river_attrs[name]) for name in river_attrs.dtype.names}
        )
        geoms = list()
        geoms = self._get_polylines(self.RIVER_CENTERLINES)
        river_gdf = GeoDataFrame(
            river_dict,
            geometry=geoms,
            crs=self.projection(),
        )
        if datetime_to_str:
            river_gdf["Last Edited"] = river_gdf["Last Edited"].apply(
                lambda x: pd.Timestamp.isoformat(x)
            )
        return river_gdf

    def flowpaths(self) -> GeoDataFrame:  # noqa D102
        raise NotImplementedError

    def bank_points(self) -> GeoDataFrame:  # noqa D102
        raise NotImplementedError

    def bank_lines(self) -> GeoDataFrame:  # noqa D102
        raise NotImplementedError

    def ineffective_areas(self) -> GeoDataFrame:  # noqa D102
        raise NotImplementedError

    def ineffective_points(self) -> GeoDataFrame:  # noqa D102
        raise NotImplementedError

    def blocked_obstructions(self) -> GeoDataFrame:  # noqa D102
        raise NotImplementedError

    def cross_sections_elevations(self, round_to: int = 2) -> pd.DataFrame:
        """Return the model cross section elevation information.

        Returns
        -------
        DataFrame
            A DataFrame containing the model cross section elevation information if they exist.
        """
        path = "/Geometry/Cross Sections"
        if path not in self:
            return pd.DataFrame()

        xselev_data = self[path]
        xs_df = self.cross_sections()
        elevations = list()
        for part_start, part_cnt in xselev_data["Station Elevation Info"][()]:
            xzdata = xselev_data["Station Elevation Values"][()][
                part_start : part_start + part_cnt
            ]
            elevations.append(xzdata)

        xs_elev_df = xs_df[
            ["xs_id", "River", "Reach", "RS", "Left Bank", "Right Bank"]
        ].copy()
        xs_elev_df["Left Bank"] = xs_elev_df["Left Bank"].round(round_to).astype(str)
        xs_elev_df["Right Bank"] = xs_elev_df["Right Bank"].round(round_to).astype(str)
        xs_elev_df["elevation info"] = elevations

        return xs_elev_df
