"""Evaluate the training data sampling using the local LISS benchmark."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys
import logging
from logging.config import dictConfig

# externals
from osgeo import gdal
import numpy as np
import xarray as xr
from sklearn.metrics import confusion_matrix, classification_report

# locals
from pysegcnn.core.utils import img2np, np2tif, array_replace
from pysegcnn.core.logging import log_conf
from pysegcnn.core.graphics import (plot_confusion_matrix,
                                    plot_classification_report)
from ai4ebv.core.legend import Legend, SUPPORTED_LC_PRODUCTS
from ai4ebv.core.dataset import HLSDataset
from ai4ebv.core.sample import TrainingDataFactory
from ai4ebv.core.cli import sample_parser
from ai4ebv.core.constants import LISS_TILES, LABELS
from ai4ebv.main.io import ROOT, TS_PATH, SAMPLE_PATH, LC_LAYERS
from ai4ebv.main.config import (YEAR, MONTHS, DROPNA, DROPQA, OVERWRITE,
                                LC_LABELS, TILES, AZURE)

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # INITIALIZE LOGGING ------------------------------------------------------
    # -------------------------------------------------------------------------
    dictConfig(log_conf())

    # define command line argument parser
    parser = sample_parser()

    # parse command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)

    # -------------------------------------------------------------------------
    # INITIALIZE LOOP OVER TILES ----------------------------------------------
    # -------------------------------------------------------------------------
    y_pred = []
    y_true = []
    p_tile = []  # list of processed tiles within extent of LISS dataset
    for tile in TILES:

        # ---------------------------------------------------------------------
        # HLS TIME SERIES -----------------------------------------------------
        # ---------------------------------------------------------------------

        # check if time series exists on disk
        time_series = TS_PATH.joinpath(
            tile, str(YEAR), HLSDataset.time_series(tile, str(YEAR), MONTHS))
        if time_series.exists() and not OVERWRITE:
            # read HLS time series from disk
            LOGGER.info('Found existing time series: {}'.format(time_series))
            hls_ts = xr.open_dataset(time_series)
        else:
            # instanciate the HLS-dataset
            hls = HLSDataset.initialize(ROOT, tile, YEAR, months=MONTHS,
                                        azure=AZURE)

            # check whether scenes are available on Azure Cloud or Nasa's Ftp
            if not hls.scenes:
                LOGGER.info('Requested tile {} for year {} is not available '
                            'on {}.'.format(hls.tile, hls.year, hls.ftp))
                continue

            # generate HLS time series: xarray.Dataset
            hls_ts = hls.to_xarray(time_series, spatial_coverage=DROPNA,
                                   cloud_coverage=DROPQA, save=True,
                                   overwrite=OVERWRITE)

        # ---------------------------------------------------------------------
        # LAND COVER LABELS ---------------------------------------------------
        # ---------------------------------------------------------------------

        # get the wte land cover layer for the current tile
        lc = LC_LAYERS[LC_LABELS][tile]
        ds = gdal.Open(str(lc))

        # ---------------------------------------------------------------------
        # TRAINING DATA SAMPLING ----------------------------------------------
        # ---------------------------------------------------------------------

        # instanciate training data factory
        factory = TrainingDataFactory(hls_ts, lc, label_name=LC_LABELS)
        samples = factory.samples.astype(np.int16)

        # training data layer
        _, training_layer = factory.generate_training_data(
            hls_ts, samples, factory.class_labels, args.npixel, features=True,
            layer_only=True, buffer=args.buffer, apriori=args.apriori)

        # ---------------------------------------------------------------------
        # SAVE OUTPUTS --------------------------------------------------------
        # ---------------------------------------------------------------------

        # output path
        tpath = SAMPLE_PATH.joinpath(tile).joinpath(str(YEAR))
        if not tpath.exists():
            tpath.mkdir(parents=True, exist_ok=True)

        # save sampling data layer
        fname = '_'.join([LC_LABELS, 'N{}'.format(args.npixel),
                          'B{}'.format(args.buffer), 'AP' if args.apriori
                          else '', 'samples.tif'])
        fname = tpath.joinpath(fname)
        np2tif(samples, fname, names=['Samples'],
               no_data=Legend.NoData.id, src_ds=ds, overwrite=True)

        # save training data layer
        fname = str(fname).replace('samples', 'training')
        np2tif(training_layer.astype(np.int16), fname, names=['Training data'],
               no_data=Legend.NoData.id, src_ds=ds, overwrite=True)

        # ---------------------------------------------------------------------
        # PREPROCESS BENCHMARK LC DATA ----------------------------------------
        # ---------------------------------------------------------------------
        if tile in LISS_TILES:
            # Benchmark land cover layer: LISS
            BNC_LAYER = LC_LAYERS['LISS'][tile]
            p_tile.append(tile)

            # benchmark land cover layer
            ref_lc = img2np(BNC_LAYER)

            # map classification legend to working legend
            ref_lc = array_replace(ref_lc,
                                   SUPPORTED_LC_PRODUCTS['LISS'].to_numpy())

            # -----------------------------------------------------------------
            # EXCLUDE NODATA VALUES -------------------------------------------
            # -----------------------------------------------------------------

            # check where the land cover datasets are defined
            defined = ((ref_lc != Legend.NoData.id) &
                       (samples != Legend.NoData.id))

            # exclude NoData values from the analysis
            y_pred.append(samples[defined])
            y_true.append(ref_lc[defined])

    # -------------------------------------------------------------------------
    # CALCULATE STATISTICS ----------------------------------------------------
    # -------------------------------------------------------------------------
    if p_tile:
        # output path for graphics
        spath = SAMPLE_PATH.joinpath('Statistics')
        if not spath.exists():
            spath.mkdir(parents=True, exist_ok=True)

        # file name for graphics
        fname = '_'.join([LC_LABELS, *p_tile, 'N{}'.format(args.npixel),
                          'B{}'.format(args.buffer), 'AP' if args.apriori
                          else ''])

        # convert labels to numpy array
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        # class names
        class_names = [v['label'] for v in LABELS.values()]

        # calculate confusion matrix: assume that the benchmark represents the
        # true state of the land cover
        LOGGER.info('Computing confusion matrix ...')
        cm = confusion_matrix(y_true, y_pred, labels=list(LABELS.keys()))

        # plot confusion matrix
        fig, _ = plot_confusion_matrix(cm, class_names, cmap='Reds',
                                       normalize=False)
        cm_name = spath.joinpath(fname + '_cm.png')
        LOGGER.info('Confusion matrix: {}'.format(cm_name))
        fig.savefig(cm_name, bbox_inches='tight', dpi=300)

        # compute classification report
        LOGGER.info('Computing classification metrics ...')
        report = classification_report(y_true, y_pred,
                                        target_names=class_names,
                                        labels=list(LABELS.keys()),
                                        zero_division=1, output_dict=True)

        # plot and save classification report
        fig = plot_classification_report(report, class_names, cmap='Reds')
        report_name = spath.joinpath(fname + '_crep.png')
        LOGGER.info('Classification report: {}'.format(report_name))
        fig.savefig(report_name, bbox_inches='tight', dpi=300)
