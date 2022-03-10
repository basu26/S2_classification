"""Convert hdf files to GeoTiff."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys
import logging
from logging.config import dictConfig

# locals
from pysegcnn.core.utils import hdf2tifs, search_files
from pysegcnn.core.logging import log_conf
from ai4ebv.core.cli import hdf_parser
from ai4ebv.core.dataset import HLSMetaDataset

# module level logger
LOGGER = logging.getLogger(__name__)

# define pattern to match for HLS dataset
HLS_PATTERN = 'HLS.(S|L)30(.*)v1.4.hdf$'


if __name__ == '__main__':

    # configure logging
    dictConfig(log_conf())

    # define command line argument parser
    parser = hdf_parser()

    # parse command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)

    # check whether the input path exists
    if args.hdfs.exists():

        # check whether the inputh path is a file or a directory
        if args.hdfs.is_dir():
            # search for files matching hls pattern in the specified directory
            files = search_files(args.hdfs, HLS_PATTERN)
        elif args.hdfs.is_file():
            files = [args.hdfs]

        # iterate over the hdf files to process
        for file in files:

            # tile and year of current scene
            tile = HLSMetaDataset.parse_tile(file.stem)
            year = HLSMetaDataset.parse_date(file.stem).year

            # output directory: outpath/tile/year
            outpath = args.outpath.joinpath(str(tile), str(year))

            # convert to GeoTiff
            hdf2tifs(file, outpath, overwrite=args.overwrite,
                     create_stack=args.stack)
    else:
        LOGGER.info('{} does not exist.'.format(str(args.hdfs)))
        sys.exit()
