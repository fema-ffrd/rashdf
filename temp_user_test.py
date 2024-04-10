import src.rashdf as rashdf
from src.rashdf import utils
from os.path import join

geom_hdf_file = r"C:\Users\USJB713989\OneDrive - WSP O365\_code\python\rashdf\data\2d\Richland_Lower.g01.hdf"
out_dir = r"C:\Users\USJB713989\OneDrive - WSP O365\Desktop\rashdf_test_out"

with rashdf.RasGeomHdf(geom_hdf_file) as ghdf:   

    from datetime import datetime
    a = datetime.now()

    ghdf.mesh_areas().to_file(join(out_dir,"mesh_areas.shp"))
    ghdf.mesh_cell_points().to_file(join(out_dir,"mesh_cell_points.shp"))
    ghdf.mesh_cell_faces().to_file(join(out_dir,"mesh_cell_faces.shp"))
    ghdf.mesh_cell_polygons().to_file(join(out_dir,"mesh_cell_polygons.shp"))

    print(datetime.now()-a)
