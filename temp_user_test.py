import src.rashdf as rashdf
from src.rashdf import utils
from os.path import join

geom_hdf_file = r"C:\Users\USJB713989\Downloads\Richland_Lower.g01.hdf"
out_dir = r"C:\Users\USJB713989\Downloads\_out"

with rashdf.RasGeomHdf(geom_hdf_file) as ghdf:    
    ghdf.mesh_areas().to_file(join(out_dir,"mesh_areas.shp"))
    ghdf.mesh_cell_points().to_file(join(out_dir,"mesh_cell_points.shp"))
    ghdf.mesh_cell_faces().to_file(join(out_dir,"mesh_cell_faces.shp"))
    ghdf.mesh_cell_polygons().to_file(join(out_dir,"mesh_cell_polygons.shp"))