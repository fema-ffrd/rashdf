import src.rashdf as rashdf
from src.rashdf import utils

with rashdf.RasGeomHdf(r"C:\Users\USJB713989\Downloads\Richland_Lower.g01.hdf") as ghdf:    
    ghdf.mesh_areas().to_file(r"C:\Users\USJB713989\Downloads\_out\mesh_areas.shp")
    ghdf.mesh_cell_points().to_file(r"C:\Users\USJB713989\Downloads\_out\mesh_cell_points.shp")
    ghdf.mesh_cell_faces().to_file(r"C:\Users\USJB713989\Downloads\_out\mesh_cell_faces.shp")
    # ghdf.mesh_cell_polygons().to_file(r"C:\Users\USJB713989\Downloads\_out\mesh_cell_polygons.shp")