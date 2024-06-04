.. rashdf documentation master file, created by
   sphinx-quickstart on Mon Jun  3 15:55:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


rashdf Documentation
====================
:code:`rashdf` is a library for reading data from `HEC-RAS <https://www.hec.usace.army.mil/software/hec-ras/>`_
HDF5 files. It is a wrapper around the :code:`h5py` library, and provides an interface with 
convenience functions for reading key HEC-RAS geometry data, output data,
and metadata.

Installation
============
With :code:`pip`::

   pip install rashdf


Quickstart
==========
Reading geometry from a HEC-RAS geometry HDF file::

   >>> from rashdf import RasGeomHdf
   >>> with RasGeomHdf('Muncie.g05.hdf') as geom_hdf:
   ...     projection = geom_hdf.projection()
   ...     mesh_area_names = geom_hdf.mesh_area_names()
   ...     mesh_areas = geom_hdf.mesh_areas()
   ...     mesh_cell_points = geom_hdf.mesh_cell_points()
   >>> projection
   <Projected CRS: EPSG:2965>
   Name: NAD83 / Indiana East (ftUS)
   Axis Info [cartesian]:
   - E[east]: Easting (US survey foot)
   - N[north]: Northing (US survey foot)
   Area of Use:
   - undefined
   Coordinate Operation:
   - name: unnamed
   - method: Transverse Mercator
   Datum: North American Datum 1983
   - Ellipsoid: GRS 1980
   - Prime Meridian: Greenwich
   >>> mesh_area_names
   ['2D Interior Area', 'Perimeter_NW']
   >>> mesh_areas
             mesh_name                                           geometry
   0  2D Interior Area  POLYGON ((409537.180 1802597.310, 409426.140 1...
   1      Perimeter_NW  POLYGON ((403914.470 1804971.220, 403008.310 1...
   >>> mesh_cell_points
                mesh_name  cell_id                        geometry
   0     2D Interior Area        0  POINT (406000.000 1805000.000)
   1     2D Interior Area        1  POINT (406050.000 1805000.000)
   2     2D Interior Area        2  POINT (406100.000 1805000.000)
   3     2D Interior Area        3  POINT (406150.000 1805000.000)
   4     2D Interior Area        4  POINT (406200.000 1805000.000)
   ...                ...      ...                             ...
   5785      Perimeter_NW      514  POINT (403731.575 1804124.860)
   5786      Perimeter_NW      515  POINT (403650.619 1804121.731)
   5787      Perimeter_NW      516  POINT (403585.667 1804141.139)
   5788      Perimeter_NW      517  POINT (403534.818 1804186.902)
   5789      Perimeter_NW      518  POINT (403632.837 1804235.708)

Reading plan data from a HEC-RAS plan HDF file hosted on AWS S3
(note, this requires installation of the optional :code:`fsspec`
and :code:`s3fs` libraries as well as configuration of S3
credentials)::

   >>> from rashdf import RasPlanHdf
   >>> with RasPlanHdf.open_uri('s3://bucket/ElkMiddle.p01.hdf') as plan_hdf:
   ...     plan_info = plan_hdf.get_plan_info_attrs()
   >>> plan_info
   {'Base Output Interval': '1HOUR', 'Computation Time Step Base': '1MIN',
   'Flow Filename': 'ElkMiddle.u01', 'Flow Title': 'Jan_1996',
   'Geometry Filename': 'ElkMiddle.g01', 'Geometry Title': 'ElkMiddle',
   'Plan Filename': 'ElkMiddle.p01', 'Plan Name': 'Jan_1996',
   'Plan ShortID': 'Jan_1996', 'Plan Title': 'Jan_1996',
   'Project Filename': 'g:\\Jan1996_Kanawha_CloudPrep\\Elk Middle\\ElkMiddle.prj',
   'Project Title': 'ElkMiddle',
   'Simulation End Time': datetime.datetime(1996, 2, 7, 12, 0),
   'Simulation Start Time': datetime.datetime(1996, 1, 14, 12, 0),
   'Time Window': [datetime.datetime(1996, 1, 14, 12, 0),
   datetime.datetime(1996, 2, 7, 12, 0)]}


API
===
.. toctree::
   :maxdepth: 1

   RasGeomHdf
   RasPlanHdf
   RasHdf

:code:`rashdf` provides two primary classes for reading data from
HEC-RAS geometry and plan HDF files: :code:`RasGeomHdf` and :code:`RasPlanHdf`.
Both of these classes inherit from the `RasHdf` base class, which
inherits from the :code:`h5py.File` class.

Note that :code:`RasPlanHdf` inherits from :code:`RasGeomHdf`, so all of the
methods available in :code:`RasGeomHdf` are also available in :code:`RasPlanHdf`.
