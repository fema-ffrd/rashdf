"""rashdf CLI."""

from rashdf import RasGeomHdf, RasPlanHdf
from rashdf.utils import df_datetimes_to_str

from geopandas import GeoDataFrame

import argparse
from ast import literal_eval
from pathlib import Path
import sys
import re
from typing import List, Optional
import warnings


COMMANDS = [
    "mesh_areas",
    "mesh_cell_points",
    "mesh_cell_polygons",
    "mesh_cell_faces",
    "refinement_regions",
    "bc_lines",
    "breaklines",
    "reference_lines",
    "reference_points",
    "structures",
]


def docstring_to_help(docstring: Optional[str]) -> str:
    """Extract the first line of a docstring to use as help text for the rashdf CLI.

    Note that this function replaces 'Return' with 'Export' in the help text.

    Parameters
    ----------
    docstring : Optional[str]
        The docstring to extract the first line from.

    Returns
    -------
    str
        The first line of the docstring with 'Return' replaced by 'Export'.
        If the docstring is None, an empty string is returned.
    """
    if docstring is None:
        return ""
    help_text = docstring.split("\n")[0]
    help_text = help_text.replace("Return", "Export")
    return help_text


def pyogrio_supported_drivers() -> List[str]:
    """Return a list of drivers supported by pyogrio for writing output files.

    Returns
    -------
    list
        A list of drivers supported by pyogrio for writing output files.
    """
    import pyogrio

    drivers = pyogrio.list_drivers(write=True)
    return sorted(drivers)


def fiona_supported_drivers() -> List[str]:
    """Return a list of drivers supported by Fiona for writing output files.

    Returns
    -------
    list
        A list of drivers supported by Fiona for writing output files.
    """
    import fiona

    drivers = [d for d, s in fiona.supported_drivers.items() if "w" in s]
    return sorted(drivers)


def parse_args(args: str) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract data from HEC-RAS HDF files.")
    parser.add_argument(
        "--pyogrio-drivers",
        action="store_true",
        help="List the drivers supported by pyogrio for writing output files.",
    )
    fiona_installed = False
    engines = ["pyogrio"]
    try:
        import fiona

        fiona_installed = True
        engines.append("fiona")
        parser.add_argument(
            "--fiona-drivers",
            action="store_true",
            help="List the drivers supported by Fiona for writing output files.",
        )
    except ImportError:
        pass
    subparsers = parser.add_subparsers(help="Sub-command help")
    for command in COMMANDS:
        f = getattr(RasGeomHdf, command)
        subparser = subparsers.add_parser(
            command, description=docstring_to_help(f.__doc__)
        )
        subparser.set_defaults(func=command)
        subparser.add_argument("hdf_file", type=str, help="Path to HEC-RAS HDF file.")
        subparser.add_argument(
            "output_file", type=str, help="Path to output file.", nargs="?"
        )
        subparser.add_argument(
            "--to-crs", type=str, help='Output CRS. (e.g., "EPSG:4326")'
        )
        output_group = subparser.add_mutually_exclusive_group()
        output_group.add_argument(
            "--parquet", action="store_true", help="Output as Parquet."
        )
        output_group.add_argument(
            "--feather", action="store_true", help="Output as Feather."
        )
        output_group.add_argument(
            "--engine",
            type=str,
            choices=engines,
            default="pyogrio",
            help="Engine for writing output data.",
        )
        subparser.add_argument(
            "--kwargs",
            type=str,
            help=(
                "Keyword arguments as a Python dictionary literal"
                " passed to the corresponding GeoPandas output method."
            ),
        )
    args = parser.parse_args(args)
    return args


def export(args: argparse.Namespace) -> Optional[str]:
    """Act on parsed arguments to extract data from HEC-RAS HDF files."""
    if args.pyogrio_drivers:
        for driver in pyogrio_supported_drivers():
            print(driver)
        return
    if hasattr(args, "fiona_drivers") and args.fiona_drivers:
        for driver in fiona_supported_drivers():
            print(driver)
        return
    if re.match(r"^.*\.p\d\d\.hdf$", args.hdf_file):
        ras_hdf_class = RasPlanHdf
    else:
        ras_hdf_class = RasGeomHdf
    if re.match(r"^\w+://", args.hdf_file):
        geom_hdf = ras_hdf_class.open_uri(args.hdf_file)
    else:
        geom_hdf = ras_hdf_class(args.hdf_file)
    func = getattr(geom_hdf, args.func)
    gdf: GeoDataFrame = func()
    kwargs = literal_eval(args.kwargs) if args.kwargs else {}
    if args.to_crs:
        gdf = gdf.to_crs(args.to_crs)
    if not args.output_file:
        # If an output file path isn't provided, write the GeoDataFrame to stdout
        # as GeoJSON. Convert any datetime columns to strings.
        gdf = df_datetimes_to_str(gdf)
        with warnings.catch_warnings():
            # Squash warnings about converting the CRS to OGC URN format.
            # Likely to come up since USACE's Albers projection is a custom CRS.
            # A warning written to stdout might cause issues with downstream processing.
            warnings.filterwarnings(
                "ignore",
                (
                    "GeoDataFrame's CRS is not representable in URN OGC format."
                    " Resulting JSON will contain no CRS information."
                ),
            )
            result = gdf.to_json(**kwargs)
        print("No output file!")
        print(result)
        return result
    elif args.parquet:
        gdf.to_parquet(args.output_file, **kwargs)
        return
    elif args.feather:
        gdf.to_feather(args.output_file, **kwargs)
        return
    output_file_path = Path(args.output_file)
    output_file_ext = output_file_path.suffix
    if output_file_ext not in [".gpkg"]:
        # Unless the user specifies a format that supports datetime,
        # convert any datetime columns to string.
        # TODO: besides Geopackage, which of the standard Fiona drivers allow datetime?
        gdf = df_datetimes_to_str(gdf)
    gdf.to_file(args.output_file, engine=args.engine, **kwargs)


def main():
    """Entry point for the rashdf CLI."""
    args = parse_args(sys.argv[1:])
    export(args)


if __name__ == "__main__":
    main()
