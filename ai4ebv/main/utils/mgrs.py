"""Tile a raster into the MGRS tiling system."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys
import logging
from logging.config import dictConfig

# locals
from pysegcnn.core.utils import raster2mgrs
from pysegcnn.core.logging import log_conf
from ai4ebv.core.cli import mgrs_parser

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # configure logging
    dictConfig(log_conf())

    # define command line argument parser
    parser = mgrs_parser()

    # parse command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)

    # check whether the input raster exists
    if args.raster.exists():

        # check whether the MGRS grid kml file exists
        if not args.grid.exists():
            LOGGER.info('MGRS grid kml {} does not exist.'.format(args.grid))
            sys.exit()

        # create output directory, if it does not exist
        if not args.outpath.exists():
            LOGGER.info('mkdir {}'.format(args.outpath))
            args.outpath.mkdir(parents=True, exist_ok=True)

        # check whether a file containing the tile names is specified
        if args.file is not None:
            with open(args.file) as f:
                tiles = [line.strip().lstrip('T') for line in f]
        else:
            tiles = [t.lstrip('T') for t in args.tiles]

        # tile raster into specified tiles
        raster2mgrs(args.raster, args.grid, tiles, args.outpath,
                    pixel_size=args.pixel_size, overwrite=args.overwrite,
                    no_data=args.nodata)

    else:
        LOGGER.info('{} does not exist.'.format(str(args.raster)))
        sys.exit()
