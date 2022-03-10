"""Compute class-wise spatial difference between classifications and LISS."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys
import logging
from logging.config import dictConfig

# externals
from osgeo import gdal

# locals
from ai4ebv.core.cli import diff2liss_parser
from ai4ebv.core.utils import spatial_class_difference
from ai4ebv.core.constants import LISS_TILES
from ai4ebv.core.landcover import WteLandCover, LC_LOOKUPS
from pysegcnn.core.utils import (img2np, array_replace, np2tif, search_files,
                                 recurse_path)
from pysegcnn.core.logging import log_conf

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # configure logging
    dictConfig(log_conf())

    # define command line argument parser
    parser = diff2liss_parser()

    # parse command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)

    # check whether a land cover product is specified
    if args.source.exists():

        # search the local benchmark land cover layer: Liss
        Liss = {tile: search_files(args.source,
                                   '^LISS(.*){}.tif$'.format(tile)).pop()
                for tile in LISS_TILES}

        # find all layers in the source directory covering the tiles of the
        # LISS dataset
        files = recurse_path(args.source)
        files = [f for f in files if (f.stem.startswith('HLS') and
                 f.stem.endswith(LISS_TILES) and f.suffix == '.tif')]

        # log found layers
        LOGGER.info('Found classified layers:')
        LOGGER.info(('\n ' + (len(__name__) + 1) * ' ').join(
                    ['{}'.format(layer) for layer in files]))

        # check whether to only list which layer would be processed
        if args.dry_run:
            sys.exit()

        # for each layer, compute the spatial difference with respect to the
        # corresponding LISS tile
        LOGGER.info('Computing class-wise difference ...')
        for layer in files:

            # read the layer to array
            ds = gdal.Open(str(layer))
            y_pred = img2np(layer)

            # get the tile of the layer and read the corresponding LISS tile
            tile = layer.stem.split('_')[-1]
            y_true = array_replace(img2np(Liss[tile]),
                                   LC_LOOKUPS['LISS'].to_numpy())

            # compute spatial class-wise difference layer
            LOGGER.info('Prediction: {}, Reference: {}'
                        .format(layer.stem, Liss[tile].stem))
            y_diff = spatial_class_difference(y_true, y_pred)

            # save difference layer
            np2tif(y_diff, filename=str(layer).replace('.tif', '_diff.tif'),
                   names=['Difference to LISS'], src_ds=ds,
                   overwrite=args.overwrite, no_data=WteLandCover.NoData.id,
                   compress=False)

    else:
        LOGGER.info('{} does not exist.'.format(str(args.source)))
        sys.exit()
