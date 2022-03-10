"""Extract HLS tiles from an archive."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys
import logging
from logging.config import dictConfig

# locals
from pysegcnn.core.logging import log_conf
from ai4ebv.core.cli import extract_parser
from ai4ebv.core.utils import extract_hls_tiles

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # initialize logging
    dictConfig(log_conf())

    # define command line argument parser
    parser = extract_parser()

    # parse command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)

    # check whether the input archive extists
    if not args.source.exists():
        LOGGER.info('{} does not exist.'.format(args.source))
        sys.exit()

    # check if the target path exists
    if not args.target.exists():
        LOGGER.info('mkdir {}'.format(args.target))
        args.target.mkdir(parents=True, exist_ok=True)

    # extract archive
    extract_hls_tiles(args.source, args.target, tiles=args.tiles,
                      year=args.year, overwrite=args.overwrite)
