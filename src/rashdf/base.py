"""Base class for reading HEC-RAS HDF files."""

import h5py
from .utils import hdf5_attrs_to_dict
from typing import Dict


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
        self._loc = name

    @classmethod
    def open_uri(
        cls, uri: str, fsspec_kwargs: dict = {}, h5py_kwargs: dict = {}
    ) -> "RasHdf":
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
        result = cls(remote_file.open(), **h5py_kwargs)
        result._loc = uri
        return result

    def get_attrs(self, attr_path: str) -> Dict:
        """Convert attributes from a HEC-RAS HDF file into a Python dictionary for a given attribute path.

        Parameters
        ----------
            attr_path (str): The path within the HEC-RAS HDF file where the desired attributes are located (Ex. "Plan Data/Plan Parameters").

        Returns
        -------
            plan_attrs (dict): Dictionary filled with attributes at given path, if attributes exist at that path.
        """
        attr_object = self.get(attr_path)

        if attr_object:
            return hdf5_attrs_to_dict(attr_object.attrs)

        return {}

    def get_root_attrs(self):
        """Return attributes at root level of HEC-RAS HDF file.

        Returns
        -------
        dict
            Dictionary filled with HEC-RAS HDF root attributes.
        """
        return self.get_attrs("/")
