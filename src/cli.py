from rashdf import RasGeomHdf

import fiona
from geopandas import GeoDataFrame

import argparse
from ast import literal_eval
import sys
from typing import List


COMMANDS = [
    "mesh_areas",
    "mesh_cell_points",
    "mesh_cell_polygons",
    "mesh_cell_faces",
    "refinement_regions",
    "bc_lines",
    "breaklines",
    "structures",
]


def docstring_to_help(docstring: str) -> str:
    """Extract the first line of a docstring to use as help text for the rashdf CLI.

    Note that this function replaces 'Return' with 'Export' in the help text.

    Parameters
    ----------
    docstring : str
        The docstring to extract the first line from.

    Returns
    -------
    str
        The first line of the docstring with 'Return' replaced by 'Export'.
    """
    help_text = docstring.split("\n")[0]
    help_text = help_text.replace("Return", "Export")
    return help_text


def fiona_supported_drivers() -> List[str]:
    """Return a list of drivers supported by Fiona for writing output files.

    Returns
    -------
    list
        A list of drivers supported by Fiona for writing output files.
    """
    drivers = [d for d, s in fiona.supported_drivers.items() if "w" in s]
    return drivers


def parse_args(args: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract data from HEC-RAS HDF files.")
    parser.add_argument(
        "--fiona-drivers",
        action="store_true",
        help="List the drivers supported by Fiona for writing output files.",
    )
    subparsers = parser.add_subparsers(help="Sub-command help")
    for command in COMMANDS:
        f = getattr(RasGeomHdf, command)
        subparser = subparsers.add_parser(
            command, description=docstring_to_help(f.__doc__)
        )
        subparser.set_defaults(func=command)
        subparser.add_argument("hdf_file", type=str, help="Path to HEC-RAS HDF file.")
        subparser.add_argument("output_file", type=str, help="Path to output file.")
        subparser.add_argument("--to-crs", type=str, help="Output CRS.")
        output_group = subparser.add_mutually_exclusive_group()
        output_group.add_argument(
            "--parquet", action="store_true", help="Output as Parquet."
        )
        output_group.add_argument(
            "--feather", action="store_true", help="Output as Feather."
        )
        output_group.add_argument(
            "--json", action="store_true", help="Output as GeoJSON."
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


def export(args: argparse.Namespace):
    if args.fiona_drivers:
        for driver in fiona_supported_drivers():
            print(driver)
        return
    if "://" in args.hdf_file:
        geom_hdf = RasGeomHdf.open_uri(args.hdf_file)
    else:
        geom_hdf = RasGeomHdf(args.hdf_file)
    func = getattr(geom_hdf, args.func)
    gdf: GeoDataFrame = func()
    kwargs = literal_eval(args.kwargs) if args.kwargs else {}
    if args.to_crs:
        gdf = gdf.to_crs(args.to_crs)
    if args.json:
        gdf.to_json(args.output_file, **kwargs)
        return
    elif args.parquet:
        gdf.to_parquet(args.output_file, **kwargs)
        return
    elif args.feather:
        gdf.to_feather(args.output_file, **kwargs)
        return
    gdf.to_file(args.output_file, **kwargs)


def main():
    args = parse_args(sys.argv[1:])
    export(args)


if __name__ == "__main__":
    main()
