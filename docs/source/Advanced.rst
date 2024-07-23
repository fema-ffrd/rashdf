Advanced
========
:code:`rashdf` provides convenience methods for generating
Zarr metadata for HEC-RAS HDF5 files. This is particularly useful
for working with stochastic  ensemble simulations, where many
HEC-RAS HDF5 files are generated for different model realizations,
forcing scenarios, or other sources of uncertainty.

To illustrate this, consider a set of HEC-RAS HDF5 files stored
in an S3 bucket, where each file represents a different simulation
of a river model. We can generate Zarr metadata for each simulation
and then combine the metadata into a single Kerchunk metadata file
that includes a new "sim" dimension. This combined metadata file
can then be used to open a single Zarr dataset that includes all
simulations.

The cell timeseries output for a single simulation might look
something like this::

    >>> from rashdf import RasPlanHdf
    >>> plan_hdf = RasPlanHdf.open_uri("s3://bucket/simulations/1/BigRiver.p01.hdf")
    >>> plan_hdf.mesh_cells_timeseries_output("BigRiverMesh1")
    <xarray.Dataset> Size: 66MB
    Dimensions:                              (time: 577, cell_id: 14188)
    Coordinates:
    * time                                 (time) datetime64[ns] 5kB 1996-01-14...
    * cell_id                              (cell_id) int64 114kB 0 1 ... 14187
    Data variables:
        Water Surface                        (time, cell_id) float32 33MB dask.array<chunksize=(3, 14188), meta=np.ndarray>
        Cell Cumulative Precipitation Depth  (time, cell_id) float32 33MB dask.array<chunksize=(3, 14188), meta=np.ndarray>
    Attributes:
        mesh_name:  BigRiverMesh1

Note that the example below requires installation of the optional
libraries :code:`kerchunk`, :code:`zarr`, :code:`fsspec`, and :code:`s3fs`::

    from rashdf import RasPlanHdf
    from kerchunk.combine import MultiZarrToZarr
    import json

    # Example S3 URL pattern for HEC-RAS plan HDF5 files
    s3_url_pattern = "s3://bucket/simulations/{sim}/BigRiver.p01.hdf"

    zmeta_files = []
    sims = list(range(1, 11))

    # Generate Zarr metadata for each simulation
    for sim in sims:
        s3_url = s3_url_pattern.format(sim=sim)
        plan_hdf = RasPlanHdf.open_uri(s3_url)
        zmeta = plan_hdf.zmeta_mesh_cells_timeseries_output("BigRiverMesh1")
        json_file = f"BigRiver.{sim}.p01.hdf.json"
        with open(json_file, "w") as f:
            json.dump(zmeta, f)
        json_list.append(json_file)
    
    # Combine Zarr metadata files into a single Kerchunk metadata file
    # with a new "sim" dimension
    mzz = MultiZarrToZarr(zmeta_files, concat_dims=["sim"], coo_map={"sim": sims})
    mzz_dict = mss.translate()

    with open("BigRiver.combined.p01.json", "w") as f:
        json.dump(mzz_dict, f)

Now, we can open the combined dataset with :code:`xarray`::

    import xarray as xr

    ds = xr.open_dataset(
        "reference://",
        engine="zarr",
        backend_kwargs={
            "consolidated": False,
            "storage_options": {"fo": "BigRiver.combined.p01.json"},
        },
        chunks="auto",
    )

The resulting combined dataset includes a new :code:`sim` dimension::

    <xarray.Dataset> Size: 674MB
    Dimensions:                              (sim: 10, time: 577, cell_id: 14606)
    Coordinates:
    * cell_id                              (cell_id) int64 117kB 0 1 ... 14605
    * sim                                  (sim) int64 80B 1 2 3 4 5 6 7 8 9 10
    * time                                 (time) datetime64[ns] 5kB 1996-01-14...
    Data variables:
        Cell Cumulative Precipitation Depth  (sim, time, cell_id) float32 337MB dask.array<chunksize=(10, 228, 14606), meta=np.ndarray>
        Water Surface                        (sim, time, cell_id) float32 337MB dask.array<chunksize=(10, 228, 14606), meta=np.ndarray>
    Attributes:
        mesh_name:  BigRiverMesh1
