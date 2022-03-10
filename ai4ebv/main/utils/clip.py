"""Clip a raster to the extent of another raster or shapefile."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys
import logging
from logging.config import dictConfig

# locals
from pysegcnn.core.utils import clip_raster, extract_by_mask, search_files
from pysegcnn.core.logging import log_conf
from ai4ebv.core.cli import clip_parser

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # configure logging
    dictConfig(log_conf())

    # define command line argument parser
    parser = clip_parser()

    # parse command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)

    # check whether the input path is a directory or a file
    if args.source.is_dir():
        # check whether a file pattern is specified
        if args.pattern is None:
            LOGGER.info('Pattern required when passing a directory.')
            sys.exit()

        # find all files matching the defined pattern
        source = sorted(search_files(args.source, args.pattern))
        if not source:
            LOGGER.info('No files matching {} in {}.'
                        .format(args.pattern, args.source))
            sys.exit()

    else:
        # check whether the input raster exists
        if args.source.exists():
            source = [args.source]
        else:
            LOGGER.info('{} does not exist.'.format(str(args.source)))
            sys.exit()

    # check target path
    if args.target is None:
        # default filename: <source-path>/<source>_clip.<suffix>
        target = [s.parent.joinpath(s.name.replace(
            s.suffix, '_clip{}'.format(s.suffix))) for s in source]
    else:
        # specified target path: <target-path>/<source>
        target = [args.target.joinpath(s.name) for s in source]

    # check if the mask raster/shapefile exists
    if not args.extent.exists():
        LOGGER.info('{} does not exist.'.format(args.extent))
        sys.exit()

    # clip rasters to extent of interest
    for src, trg in zip(source, target):
        # clip by shapefile
        if args.extent.suffix == '.shp':
            extract_by_mask(src, args.extent, trg, overwrite=args.overwrite,
                            src_no_data=args.source_nodata,
                            trg_no_data=args.target_nodata)
        # clip by raster
        else:
            clip_raster(src, args.extent, trg, buffer=args.buffer,
                        overwrite=args.overwrite,
                        src_no_data=args.source_nodata,
                        trg_no_data=args.target_nodata)
