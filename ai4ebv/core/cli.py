"""Command line argument parsers."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import argparse
import logging
import pathlib

# locals
from ai4ebv.core.landcover import LC_DATASET_NAMES

# epilogue to display at the end of each parser
EPILOGUE = 'Author: Daniel Frisinghelli, daniel.frisinghelli@gmail.com'

# module level logger
LOGGER = logging.getLogger(__name__)

# default values
DEFAULT = '(default: %(default)s)'


# parser to convert hdf files to GeoTiff: ai4ebv.main.utils.hls2tif.py
def hdf_parser():

    # define command line argument parser
    parser = argparse.ArgumentParser(
        description='Convert a .hdf file to GeoTiff.',
        epilog=EPILOGUE,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=50, indent_increment=2))

    # positional arguments

    # positional argument: path to the hdf files to convert
    parser.add_argument('hdfs', type=pathlib.Path,
                        help='Path to search for HLS hdf files.')

    # positional argument: path to save the GeoTIFF files
    parser.add_argument('outpath', type=pathlib.Path,
                        help='Path to save GeoTIFF files.')

    # optional arguments

    # optional argument: whether to create a GeoTIFF stack
    parser.add_argument('-s', '--stack', type=bool,
                        help='Create a GeoTIFF stack {}.'.format(DEFAULT),
                        default=False, nargs='?', const=True, metavar='')

    # optional argument: whether to overwrite existing GeoTIFF files
    parser.add_argument('-o', '--overwrite', type=bool,
                        help='Overwrite GeoTIFF files {}.'.format(DEFAULT),
                        default=False, nargs='?', const=True, metavar='')

    return parser


# parser to convert hdf files to GeoTiff: ai4ebv.main.utils.mgrs.py
def mgrs_parser():

    # define command line argument parser
    parser = argparse.ArgumentParser(
        description='Tile a raster into the MGRS tiling system.',
        epilog=EPILOGUE,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=50, indent_increment=2))

    # positional arguments

    # positional argument: path to the raster to tile
    parser.add_argument('raster', type=pathlib.Path,
                        help='Path to the raster dataset.')

    # positional argument: path to the MGRS grid kml
    parser.add_argument('grid', type=pathlib.Path,
                        help='Path to the MGRS grid kml file.')

    # positional argument: path to save the tiles
    parser.add_argument('outpath', type=pathlib.Path,
                        help='Path to save the tiles.')

    # optional arguments

    # optional argument: names of the tiles to process: from cli
    parser.add_argument('-t', '--tiles', type=str,
                        help='Names of the tiles {}.'.format(DEFAULT),
                        default=['T32TPS'], nargs='+', metavar='')

    # optional argument: names of the tiles to process: from file
    parser.add_argument('-f', '--file', type=pathlib.Path,
                        help=('Newline-delimited file containing the names of '
                              ' the tiles {}.'.format(DEFAULT)), default=None,
                        metavar='')

    # optional argument: output raster resolution
    parser.add_argument('-p', '--pixel_size', type=int,
                        help=('Spatial resolution: x_res, y_res. {}.'
                              .format(DEFAULT)),
                        default=(None, None), nargs=2, metavar='')

    # optional argument: NoData value
    parser.add_argument('-n', '--nodata', type=int,
                        help=('NoData value {}.'.format(DEFAULT)),
                        default=None, metavar='')

    # optional argument: whether to overwrite existing tiles
    parser.add_argument('-o', '--overwrite', type=bool,
                        help='Overwrite existing tiles {}.'.format(DEFAULT),
                        default=False, nargs='?', const=True, metavar='')

    return parser


# parser to create a mosaic of GeoTiff files: ai4ebv.main.utils.mosaic.py
def mosaic_parser():

    # define command line argument parser
    parser = argparse.ArgumentParser(
        description='Mosaic images matching a pattern in a directory.',
        epilog=EPILOGUE,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=50, indent_increment=2))

    # positional arguments

    # positional argument: path to the directory to search for files
    parser.add_argument('source', type=pathlib.Path,
                        help='The directory to search for images.')

    # positional argument: path to save the mosaic GeoTIFF
    parser.add_argument('target', type=pathlib.Path,
                        help='Path to save mosaic.')

    # positional argument: pattern to match
    parser.add_argument('pattern', type=str,
                        help='Pattern to search for in source.')

    # optional arguments

    # optional argument: target coordinate reference system
    parser.add_argument('-c', '--crs', type=int,
                        help='Coordinate reference system {}.'.format(DEFAULT),
                        default=4326, metavar='')

    # optional argument: target NoData value
    parser.add_argument('-n', '--nodata', type=int,
                        help=('NoData value {}.'.format(DEFAULT)),
                        default=255, metavar='')

    # optional argument: whether to overwrite existing GeoTIFF files
    parser.add_argument('-o', '--overwrite', type=bool,
                        help='Overwrite mosaic GeoTIFF {}.'.format(DEFAULT),
                        default=False, nargs='?', const=True, metavar='')

    return parser


# parser to clip a raster to extent of interest: ai4ebv.main.utils.clip.py
def clip_parser():

    # define command line argument parser
    parser = argparse.ArgumentParser(
        description='Clip a raster to a defined extent.',
        epilog=EPILOGUE,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=50, indent_increment=2))

    # positional arguments

    # positional argument: path to the directory to search for files
    parser.add_argument('source', type=pathlib.Path,
                        help='The raster(s) to clip.')

    # positional argument: path to the raster or shapefile defining the extent
    parser.add_argument('extent', type=pathlib.Path,
                        help='The raster or shapefile defining the extent.')

    # optional arguments

    # optional argument: path to save the mosaic GeoTIFF
    parser.add_argument('-t', '--target', type=pathlib.Path,
                        help=('Path to save clipped raster(s) {}.'
                              .format(DEFAULT)), default=None, metavar='')

    # optional argument: buffer size
    parser.add_argument('-b', '--buffer', type=float,
                        help=('Optional amount of buffering {}.'
                              .format(DEFAULT)),
                        default=None, metavar='')

    # optional argument: source NoData value
    parser.add_argument('-p', '--pattern', type=str,
                        help=('File pattern to search {}.'.format(DEFAULT)),
                        default=None, metavar='')

    # optional argument: source NoData value
    parser.add_argument('-sn', '--source-nodata', type=int,
                        help=('NoData value {}.'.format(DEFAULT)),
                        default=None, metavar='')

    # optional argument: target NoData value
    parser.add_argument('-tn', '--target-nodata', type=int,
                        help=('NoData value {}.'.format(DEFAULT)),
                        default=None, metavar='')

    # optional argument: whether to overwrite existing GeoTIFF files
    parser.add_argument('-o', '--overwrite', type=bool,
                        help=('Overwrite clipped raster, if it exists {}.'
                              .format(DEFAULT)),
                        default=False, nargs='?', const=True, metavar='')

    return parser


# parser to evaluate model predictions: ai4ebv.main.utils.eval.py
def eval_parser():

    # define command line argument parser
    parser = argparse.ArgumentParser(
        description='Calculate agreement between land cover products.',
        epilog=EPILOGUE,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=50, indent_increment=2))

    # positional arguments

    # positional argument: path to the raster to evaluate
    parser.add_argument('source', type=pathlib.Path,
                        help='Path to the raster to evaluate.')

    # positional argument: path to the reference raster
    parser.add_argument('target', type=pathlib.Path,
                        help=('Path to the reference raster. Has to cover the '
                              'same spatial extent as the source raster.'))

    # optional arguments

    # optional argument: name of the source land cover product
    parser.add_argument('-s', '--source-name', type=str,
                        help=('Name of the source land cover product {}.'
                              .format(DEFAULT)), default=None, metavar='',
                        choices=LC_DATASET_NAMES)

    # optional argument: name of the target land cover product
    parser.add_argument('-t', '--target-name', type=str,
                        help=('Name of the source land cover product {}.'
                              .format(DEFAULT)), default=None, metavar='',
                        choices=LC_DATASET_NAMES)

    return parser


# parser to evaluate model predictions: ai4ebv.main.utils.eval_lucas.py
def eval_lucas_parser():

    # define command line argument parser
    parser = argparse.ArgumentParser(
        description='Evaluate a land cover product against the LUCAS survey.',
        epilog=EPILOGUE,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=50, indent_increment=2))

    # positional arguments

    # positional argument: path to the model prediction
    parser.add_argument('source', type=pathlib.Path,
                        help='The land cover product. A raster layer.')

    # positional argument: path to the model prediction
    parser.add_argument('lucas', type=pathlib.Path,
                        help='The Lucas dataset. A NetCDF file.')

    # optional arguments

    # optional argument: name of the source land cover product
    parser.add_argument('-n', '--name', type=str,
                        help=('Name of the source land cover product {}.'
                              .format(DEFAULT)), default=None, metavar='',
                        choices=LC_DATASET_NAMES)

    # optional argument: whether to save the pixels covered by the LUCAS survey
    parser.add_argument('-s', '--save', type=str,
                        help=('Save the pixels covered by the LUCAS survey {}.'
                              .format(DEFAULT)), default=False, metavar='',
                        nargs='?', const=True)

    return parser


# parser to sample training data: ai4ebv.main.utils.sample.py
def sample_parser():

    # define command line argument parser
    parser = argparse.ArgumentParser(
        description='Training data sampling.',
        epilog=EPILOGUE,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=50, indent_increment=2))

    # positional arguments

    # positional argument: number of pixels to sample
    parser.add_argument('npixel', type=int, help='Number of pixels to sample.')

    # positional argument: number of pixels to sample
    parser.add_argument('buffer', type=int, help='Pixel buffer size.')

    # optional arguments

    # optional argument: number of sampled pixels based on apriori distribution
    parser.add_argument('-p', '--apriori', type=bool,
                        help=('The number of pixels for each class is '
                              'inferred using the apriori distribution. {}.'
                              .format(DEFAULT)), default=False, nargs='?',
                        const=True, metavar='')
    return parser


# parser to convert a land cover product to WTE: ai4ebv.main.utils.lc2wte.py
def wte_parser():

    # define command line argument parser
    parser = argparse.ArgumentParser(
        description='Convert a land cover product to the WTE legend.',
        epilog=EPILOGUE,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=50, indent_increment=2))

    # positional arguments

    # positional argument: path to the land cover product to convert
    parser.add_argument('product', type=pathlib.Path,
                        help='Path to the land cover product to convert.')

    # optional arguments

    # optional argument: the name of the land cover product
    parser.add_argument('-n', '--name', type=str,
                        help=('Name of the land cover product {}.'
                              .format(DEFAULT)), default=None,
                        choices=LC_DATASET_NAMES)

    # optional argument: batch mode,
    parser.add_argument('-b', '--batch', type=bool,
                        help=('Read layers to process from ai4ebv.main.io.py '
                              '{}.'.format(DEFAULT)), default=False, nargs='?',
                        const=True, metavar='')

    return parser


# parser to compute difference w.r.t. LISS: ai4ebv.main.utils.diff2liss.py
def diff2liss_parser():

    # define command line argument parser
    parser = argparse.ArgumentParser(
        description=('Compute class-wise spatial difference between '
                     'classifications and LISS.'),
        epilog=EPILOGUE,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=50, indent_increment=2))

    # positional arguments

    # positional argument: path to the land cover product to convert
    parser.add_argument('source', type=pathlib.Path,
                        help='Path to the classified land cover layers.')

    # optional arguments

    # optional argument: dry-run
    parser.add_argument('-d', '--dry-run', type=str,
                        help=('If specified, only list which layers would be '
                              'processed {}.'.format(DEFAULT)), default=False,
                        nargs='?', const=True, metavar='')

    # optional argument: whether to overwrite existing GeoTIFF files
    parser.add_argument('-o', '--overwrite', type=bool,
                        help=('Overwrite existing difference layers {}.'
                              .format(DEFAULT)),
                        default=False, nargs='?', const=True, metavar='')

    return parser


# parser to convert WTE-EC to WTE-LC: ai4ebv.main.utils.wte2lc.py
def wte_eco2lc_parser():

    # define command line argument parser
    parser = argparse.ArgumentParser(
        description=('Convert the World Terrestrial Ecosystems to the land '
                     ' cover component.'),
        epilog=EPILOGUE,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=50, indent_increment=2))

    # positional arguments

    # positional argument: path to the land cover product to convert
    parser.add_argument('source', type=pathlib.Path,
                        help='Path to the World Terrestrial Ecosystem layer.')

    # positional argument: path to the World Terrestrial Ecosystem attribute
    # table
    parser.add_argument('table', type=pathlib.Path,
                        help=('Path to the World Terrestrial Ecosystem '
                              'attribute table.'))

    # positional argument: value representing surface water
    parser.add_argument('water', type=int,
                        help='Value representing surface water.')

    # optional arguments

    # optional argument: whether to overwrite existing GeoTIFF files
    parser.add_argument('-o', '--overwrite', type=bool,
                        help=('Overwrite existing WTE land cover layers {}.'
                              .format(DEFAULT)),
                        default=False, nargs='?', const=True, metavar='')

    return parser


# parser to extract HLS tiles from an archive: ai4ebv.main.utils.extract.py
def extract_parser():

    # define command line argument parser
    parser = argparse.ArgumentParser(
        description=('Extract HLS tiles from an archive.'),
        epilog=EPILOGUE,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=50, indent_increment=2))

    # positional arguments

    # positional argument: path to the land cover product to convert
    parser.add_argument('source', type=pathlib.Path,
                        help='Path to the archive to extract.')

    # positional argument: path to the World Terrestrial Ecosystem attribute
    # table
    parser.add_argument('target', type=pathlib.Path,
                        help=('Path to save the extracted tiles.'))

    # optional arguments

    # optional argument: name of the tile to extract
    parser.add_argument('-t', '--tiles', type=str,
                        help='Name of the tiles to extract {}.'.format(DEFAULT),
                        default=None, nargs='+', metavar='')

    # optional argument: Name of the year to extract
    parser.add_argument('-y', '--year', type=int,
                        help='Year to extract {}.'.format(DEFAULT),
                        default=None, metavar='')

    # optional argument: whether to overwrite extracted files
    parser.add_argument('-o', '--overwrite', type=bool,
                        help=('Whether to overwrite extracted files {}.'
                              .format(DEFAULT)),
                        default=False, nargs='?', const=True, metavar='')

    return parser


# parser to convert Hammond Landforms to WTE-LF: ai4ebv.main.utils.hammond.py
def hammond_parser():

    # define command line argument parser
    parser = argparse.ArgumentParser(
        description=('Convert the Hammond Landforms to the World Terrestrial '
                     'Ecosystems landform component.'),
        epilog=EPILOGUE,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=50, indent_increment=2))

    # positional arguments

    # positional argument: path to the Hammond landforms to convert
    parser.add_argument('source', type=pathlib.Path,
                        help='Path to the Hammond landforms layer.')

    # optional arguments

    # optional argument: whether to overwrite existing GeoTIFF files
    parser.add_argument('-o', '--overwrite', type=bool,
                        help=('Overwrite existing WTE landform layers {}.'
                              .format(DEFAULT)),
                        default=False, nargs='?', const=True, metavar='')

    return parser
