"""Convert the World Terrestrial Ecosystems to the land cover component."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys
import logging
from logging.config import dictConfig

# externals
import numpy as np
import pandas as pd
from osgeo import gdal

# locals
from pysegcnn.core.utils import img2np, array_replace, np2tif
from pysegcnn.core.logging import log_conf
from ai4ebv.core.cli import wte_eco2lc_parser
from ai4ebv.core.landcover import WteLandCover

# module level logger
LOGGER = logging.getLogger(__name__)

# value to assign to NoData in each of the land cover products
NODATA = WteLandCover.NoData.id


if __name__ == '__main__':

    # initialize logging
    dictConfig(log_conf())

    # define command line argument parser
    parser = wte_eco2lc_parser()

    # parse command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)

    # check whether the input path exists
    if not args.source.exists():
        LOGGER.info('{} does not exist.'.format(str(args.source)))
        sys.exit()
    else:
        # check if the attribute table exists
        if not args.table.exists():
            LOGGER.info('{} does not exist.'.format(str(args.table)))
            sys.exit()

        # read the World Terrestrial Ecosystems attribute table to DataFrame
        df = pd.read_csv(str(args.table), delimiter=';')

        # the lookup table defining the mapping from the WTE Ecosystem
        # identifier to the land cover component
        lt = df.loc[:, df.columns.isin(['RealmWE_ID','LC_Class'])].to_numpy()

        # add mapping for surface water
        lt = np.append(lt, [[args.water, WteLandCover.Water.id]], axis=0)

        # read WTE layer and convert to land cover component
        lc = array_replace(img2np(args.source), lt).astype(np.int16)

        # save land cover component to disk
        filename = str(args.source).replace(args.source.suffix,
                                            '_LC{}'.format(args.source.suffix))
        np2tif(lc, filename, no_data=WteLandCover.NoData.id,
               overwrite=args.overwrite, src_ds=gdal.Open(str(args.source)),
               compress=False)
