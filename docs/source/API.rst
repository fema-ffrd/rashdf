API
===
.. toctree::
   :maxdepth: 1

   RasGeomHdf
   RasPlanHdf
   RasHdf

:code:`rashdf` provides two primary classes for reading data from
HEC-RAS geometry and plan HDF files: :code:`RasGeomHdf` and :code:`RasPlanHdf`.
Both of these classes inherit from the :code:`RasHdf` base class, which
inherits from the :code:`h5py.File` class.

Note that :code:`RasPlanHdf` inherits from :code:`RasGeomHdf`, so all of the
methods available in :code:`RasGeomHdf` are also available in :code:`RasPlanHdf`.