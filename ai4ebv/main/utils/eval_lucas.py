"""Evaluate a land cover product using LUCAS survey as benchmark."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys
import logging
from logging.config import dictConfig

# externals
import numpy as np
import xarray as xr
from osgeo import gdal
from sklearn.metrics import confusion_matrix

# locals
from pysegcnn.core.utils import (img2np, extract_by_points, np2tif,
                                 array_replace)
from pysegcnn.core.logging import log_conf
from pysegcnn.core.graphics import (plot_confusion_matrix,
                                    plot_classification_report)
from ai4ebv.core.landcover import WteLandCover, LC_LOOKUPS
from ai4ebv.core.constants import LABELS, CLASS_LABELS
from ai4ebv.core.metrics import area_adjusted_classification_report
from ai4ebv.core.cli import eval_lucas_parser
from ai4ebv.main.io import GRAPHICS_PATH

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # INITIALIZE LOGGING ------------------------------------------------------
    # -------------------------------------------------------------------------
    dictConfig(log_conf())

    # define command line argument parser
    parser = eval_lucas_parser()

    # parse command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)

    # check whether the root directory exists
    if args.source.exists():

        # generate output filename for graphics
        fname = GRAPHICS_PATH.joinpath(args.source.stem + '_lucas')

        # check if the LUCAS survey dataset exists
        if not args.lucas.exists():
            LOGGER.info('{} does not exist.'.format(str(args.lucas)))
            sys.exit()

        # read LUCAS survey dataset
        LOGGER.info('Reading Lucas dataset: {}'.format(args.lucas))
        Lucas = xr.open_dataset(args.lucas)

        # get the LUCAS points which are located in the reference layers
        LOGGER.info('Extracting LUCAS points within raster: {}'
                    .format(args.source))
        points, rows, cols = extract_by_points(
            args.source, Lucas.lon.values, Lucas.lat.values)

        # get the indices of the Lucas records within the layer
        indices = []
        for point in points:
            indices.append(np.where((Lucas.lon.values == point[0]) &
                                    (Lucas.lat.values == point[1]))[0].item())
        y_true = Lucas.LCWTE_Letter.sel(record=indices).values.astype(str)

        # check the input land cover product
        LOGGER.info('Reading land cover product ...')
        y_p = img2np(args.source)
        if args.name is not None:
            LOGGER.info('Land cover product: {}'.format(args.name))
            y_p = array_replace(y_p, LC_LOOKUPS[args.name].to_numpy())

        # save layer of selected LUCAS pixels within raster
        if args.save:
            selected_pixels = (np.ones(shape=y_p.shape, dtype=np.int16) *
                               WteLandCover.NoData.id)
            selected_pixels[rows, cols] = y_p[rows, cols]
            np2tif(selected_pixels, filename=str(fname) + '.tif',
                   no_data=WteLandCover.NoData.id, overwrite=True,
                   src_ds=gdal.Open(str(args.source)))

        # subset model predictions to LUCAS points
        y_pred = y_p[rows, cols]

        # replace missing values
        y_true[np.where(y_true == 'NA')] = WteLandCover.NoData.id
        y_true[np.where(y_true == '')] = WteLandCover.NoData.id

        # convert to integer
        y_true = y_true.astype(np.int16)

        # check where both the reference layer and the Lucas dataset are
        # defined
        defined = ((y_true != WteLandCover.NoData.id) &
                   (y_pred != WteLandCover.NoData.id))

        # exclude NoData values from the evaluation
        y_pred = y_pred[defined]
        y_true = y_true[defined]

        # calculate confusion matrix: assume that the benchmark represents
        # the true state of the land cover
        LOGGER.info('Computing confusion matrix ...')
        cm = confusion_matrix(y_true, y_pred, labels=list(LABELS.keys()),
                              normalize=None)

        # plot confusion matrix
        fig, _ = plot_confusion_matrix(cm, CLASS_LABELS, cmap='Greens',
                                       normalize=False)
        cm_name = str(fname) + '_cm.png'
        LOGGER.info('Confusion matrix: {}'.format(cm_name))
        fig.savefig(cm_name, dpi=300, bbox_inches='tight')

        # compute classification report
        LOGGER.info('Computing classification metrics ...')
        report = area_adjusted_classification_report(
            y_true, y_pred, y_p, labels=list(LABELS.keys()),
            target_names=CLASS_LABELS)

        # plot and save classification report
        fig = plot_classification_report(report, CLASS_LABELS, cmap='Greens')
        report_name = str(fname) + '_crep.png'
        LOGGER.info('Classification report: {}'.format(report_name))
        fig.savefig(report_name, bbox_inches='tight', dpi=300)

    else:
        LOGGER.info('{} does not exist.'.format(str(args.source)))
        sys.exit()
