"""Convert land cover layers to WTE legend."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys
import logging
from logging.config import dictConfig

# externals
from osgeo import gdal
import numpy as np

# locals
from ai4ebv.core.cli import wte_parser
from ai4ebv.core.landcover import WteLandCover, LC_LOOKUPS
from pysegcnn.core.utils import img2np, array_replace, np2tif
from pysegcnn.core.logging import log_conf

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # configure logging
    dictConfig(log_conf())

    # define command line argument parser
    parser = wte_parser()

    # parse command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)

    # check whether a land cover product is specified
    if args.batch:

        # tile-based conversion to WTE-legend
        from ai4ebv.main.io import LC_LAYERS

        # iterate over the different land cover products
        for product, layers in LC_LAYERS.items():
               # skip WTE-layers
               if product == 'WTE':
                   continue

               # iterate over the different tiles
               for tile, layer in layers.items():
                   # read land cover product and convert to WTE-legend
                   ds = gdal.Open(str(layer))
                   lc = array_replace(img2np(layer),
                                      LC_LOOKUPS[product].to_numpy())

                   # save layer to disk
                   np2tif(lc.astype(np.int16),
                          str(layer).replace('.tif', '_wte.tif'),
                          no_data=WteLandCover.NoData.id,
                          names='WTE Land Cover', src_ds=ds, overwrite=True,
                          compress=False)

    else:
        # check whether the input raster exists
        if args.product.exists():

            # check whether the name of the product is specified
            if args.name is None:
                LOGGER.info('Specify the name of the land cover product!')
                sys.exit()

            # read land cover product and convert to WTE-legend
            ds = gdal.Open(str(args.product))
            lc = array_replace(img2np(args.product),
                               LC_LOOKUPS[args.name].to_numpy())

            # save layer to disk
            np2tif(lc.astype(np.int16),
                   str(args.product).replace('.tif', '_wte.tif'),
                   no_data=WteLandCover.NoData.id, names='WTE Land Cover',
                   src_ds=ds, overwrite=True, compress=False)

        else:
            LOGGER.info('{} does not exist.'.format(str(args.product)))
            sys.exit()
