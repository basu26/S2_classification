"""Aggregate the Hammond Landforms legend."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys
import logging
from logging.config import dictConfig
from collections.abc import Iterable

# externals
import numpy as np
from osgeo import gdal

# locals
from pysegcnn.core.utils import img2np, array_replace, np2tif
from pysegcnn.core.logging import log_conf
from ai4ebv.core.cli import hammond_parser
from ai4ebv.core.landforms import (LandformAggregation, SayreLandforms,
                                   HammondLandforms)

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # initialize logging
    dictConfig(log_conf())

    # define command line argument parser
    parser = hammond_parser()

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
        # get and replace the NoData value
        ds = gdal.Open(str(args.source))
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        lookup = np.append(LandformAggregation.to_numpy(),
                           [[nodata, SayreLandforms.NoData.id]], axis=0)

        # read the Hammond landforms layer
        hlf = img2np(args.source)

        # replace values that are not defined by NoData
        defined = []
        for k in HammondLandforms.label_dict().keys():
            defined.extend(k) if isinstance(k, Iterable) else defined.append(k)
        hlf[~np.isin(hlf, defined)] = SayreLandforms.NoData.id

        # convert to WTE landforms
        lf = array_replace(hlf, lookup)

        # save landform component to disk
        filename = str(args.source).replace(
            args.source.suffix, '_WTE_LF{}'.format(args.source.suffix))
        np2tif(lf, filename, no_data=SayreLandforms.NoData.id,
               overwrite=args.overwrite, src_ds=ds, compress=False)
