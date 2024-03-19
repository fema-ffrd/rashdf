import h5py


class RasHdf(h5py.File):
    """Base class for reading RAS HDF files."""

    def __init__(self, name: str, **kwargs):
        """Open a HEC-RAS HDF file.
        
        Parameters
        ----------
        name : str
            The path to the RAS HDF file.
        kwargs : dict
            Additional keyword arguments to pass to h5py.File
        """
        super().__init__(name, mode="r", **kwargs)

    @classmethod
    def open_uri(cls, uri: str, fsspec_kwargs: dict = {}, h5py_kwargs: dict = {}) -> 'RasHdf':
        """Open a HEC-RAS HDF file from a URI.

        Parameters
        ----------
        uri : str
            The URI of the RAS HDF file. Note this should be a path
            recognized by fsspec, such as an S3 path or a Google Cloud
            Storage path. See https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.open
        fsspec_kwargs : dict
            Additional keyword arguments to pass to fsspec.open
        h5py_kwargs : dict
            Additional keyword arguments to pass to h5py.File

        Returns
        -------
        RasHdf
            The RAS HDF file opened from the URI.

        Examples
        --------
        >>> results_hdf = RasHdf.open_uri("s3://my-bucket/results.hdf")
        """
        import fsspec
        remote_file = fsspec.open(uri, mode="rb", **fsspec_kwargs)
        return cls(remote_file.open(), **h5py_kwargs)