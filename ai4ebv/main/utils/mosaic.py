"""Create a mosaic of GeoTiff files matching a defined pattern."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys
import logging
from logging.config import dictConfig

# locals
from pysegcnn.core.utils import search_files
from pysegcnn.core.logging import log_conf
from ai4ebv.core.cli import mosaic_parser
from ai4ebv.core.utils import mosaic_tiles

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # configure logging
    dictConfig(log_conf())

    # define command line argument parser
    parser = mosaic_parser()

    # parse command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)

    # check whether the input path exists
    if args.source.exists():

        # search the source directory for files matching the specified pattern
        tifs = search_files(args.source, args.pattern)

        # create mosaic
        mosaic_tiles(tifs, targets=args.target, trg_crs=args.crs,
                     no_data=args.nodata)

    else:
        LOGGER.info('{} does not exist.'.format(str(args.source)))
        sys.exit()
