# rashdf
[![CI](https://github.com/fema-ffrd/rashdf/actions/workflows/continuous-integration.yml/badge.svg?branch=main)](https://github.com/fema-ffrd/rashdf/actions/workflows/continuous-integration.yml)
[![Release](https://github.com/fema-ffrd/rashdf/actions/workflows/release.yml/badge.svg)](https://github.com/fema-ffrd/rashdf/actions/workflows/release.yml)
[![PyPI version](https://badge.fury.io/py/rashdf.svg)](https://badge.fury.io/py/rashdf)
[![codecov](https://codecov.io/gh/fema-ffrd/rashdf/graph/badge.svg?token=CTIIONEHV1)](https://codecov.io/gh/fema-ffrd/rashdf)
[![Documentation Status](https://readthedocs.org/projects/rashdf/badge/?version=latest)](https://rashdf.readthedocs.io/en/latest/?badge=latest)

Read data from [HEC-RAS](https://www.hec.usace.army.mil/software/hec-ras/) [HDF](https://github.com/HDFGroup/hdf5) files.

*Pronunciation: `raz·aitch·dee·eff`*

## Documentation
[rashdf on ReadTheDocs](http://rashdf.readthedocs.io/)

## Install
```bash
$ pip install rashdf
```

## Usage
`RasGeomHdf` and `RasPlanHdf` are extensions of
[h5py.File](https://docs.h5py.org/en/stable/high/file.html#h5py.File). They contain
methods to export HEC-RAS model geometry as
[GeoDataFrame](https://geopandas.org/en/stable/docs/reference/geodataframe.html)
objects.
```python
>>> from rashdf import RasGeomHdf
>>> geom_hdf = RasGeomHdf("path/to/rasmodel/Muncie.g04.hdf")
>>> mesh_cells = geom_hdf.mesh_cell_polygons()  # export a GeoDataFrame
>>> mesh_cells
             mesh_name  cell_id                                           geometry
0     2D Interior Area        0  POLYGON ((406025.000 1805015.237, 406025.000 1...
1     2D Interior Area        1  POLYGON ((406075.000 1805018.545, 406075.000 1...
2     2D Interior Area        2  POLYGON ((406075.000 1804975.000, 406075.000 1...
3     2D Interior Area        3  POLYGON ((406125.000 1804975.000, 406125.000 1...
4     2D Interior Area        4  POLYGON ((406175.000 1804975.000, 406175.000 1...
...                ...      ...                                                ...
5386  2D Interior Area     5386  POLYGON ((409163.402 1802463.621, 409175.000 1...
5387  2D Interior Area     5387  POLYGON ((409160.953 1802374.120, 409125.000 1...
5388  2D Interior Area     5388  POLYGON ((409163.402 1802463.621, 409161.906 1...
5389  2D Interior Area     5389  POLYGON ((409112.480 1802410.114, 409112.046 1...
5390  2D Interior Area     5390  POLYGON ((409112.480 1802410.114, 409063.039 1...
>>> mesh_cells.to_file("mucie-mesh-cell-polygons.shp")
```

Also, methods to extract certain HDF group attributes as dictionaries:
```python
>>> from rashdf import RasPlanHdf
>>> with RasPlanHdf("path/to/rasmodel/Muncie.p04.hdf") as plan_hdf:
>>> results_unsteady_summary_attrs = plan_hdf.get_results_unsteady_summary_attrs()
>>> results_unsteady_summary_attrs
{'Computation Time DSS': datetime.timedelta(0),
'Computation Time Total': datetime.timedelta(seconds=23),
'Maximum WSEL Error': 0.0099277812987566,
'Maximum number of cores': 6,
'Run Time Window': [datetime.datetime(2024, 3, 27, 9, 31, 52),
datetime.datetime(2024, 3, 27, 9, 32, 15)],
'Solution': 'Unsteady Finished Successfully',
'Time Solution Went Unstable': None,
'Time Stamp Solution Went Unstable': 'Not Applicable'}
```

## CLI
The `rashdf` command-line interface allows export directly to a variety of formats, enabled
by GeoPandas.
```
$ rashdf <sub-command> <hdf-file> [<output-path>] [<options>]
```

CLI help:
```
$ rashdf --help
```

Print the output formats supported by pyorgio:
```
$ rashdf --pyogrio-drivers
```

Help for a specific subcommand:
```
$ rashdf mesh_cell_polygons --help
```

Example: export mesh cell faces to an ESRI Shapefile
```
$ rashdf mesh_cell_faces BigRiver.g01.hdf big-river-mesh-cell-faces.shp
```

Example: export mesh cell points to GeoParquet
```
$ rashdf mesh_cell_points LittleCreek.g01.hdf --parquet little-creek-mesh-cell-points.parquet
```

Example: export breaklines to OGC GeoPackage and reproject to a different CRS
```
$ rashdf breaklines Whitemarsh.p01.hdf whitemarsh-breaklines.gpkg --to-crs EPSG:4326
```

Example: write structures GeoJSON to `stdout`:
```
$ rashdf structures Potomac.p01.hdf
```

## Developer Setup
Create a virtual environment in the project directory:
```
$ python -m venv venv-rashdf
```

Activate the virtual environment:
```
# For macOS/Linux
$ source ./venv-rashdf/bin/activate
(venv-rashdf) $

# For Windows
> ./venv-rashdf/Scripts/activate
```

Install dev dependencies:
```
(venv-rashdf) $ pip install ".[dev]"
```

Install git hook scripts (used for automatic liniting/formatting)
```
(venv-rashdf) $ pre-commit install
```

With the virtual environment activated, run the tests:
```
(venv-rashdf) $ pytest
```


