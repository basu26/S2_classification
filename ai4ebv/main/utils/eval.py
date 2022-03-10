"""Compare different land cover products."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys
import logging
from logging.config import dictConfig

# externals
from sklearn.metrics import confusion_matrix

# locals
from pysegcnn.core.utils import img2np, array_replace
from pysegcnn.core.logging import log_conf
from pysegcnn.core.graphics import (plot_confusion_matrix,
                                    plot_classification_report)
from ai4ebv.core.landcover import WteLandCover, LC_LOOKUPS
from ai4ebv.core.constants import LABELS, CLASS_LABELS
from ai4ebv.core.cli import eval_parser
from ai4ebv.core.metrics import area_adjusted_classification_report
from ai4ebv.main.io import GRAPHICS_PATH

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # INITIALIZE LOGGING ------------------------------------------------------
    # -------------------------------------------------------------------------
    dictConfig(log_conf())

    # define command line argument parser
    parser = eval_parser()

    # parse command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)

    # check whether the root directory exists
    if args.source.exists() and args.target.exists():

        # read input rasters
        y_pred = img2np(args.source)
        y_true = img2np(args.target)

        # check if source dataset name is specified and apply corresponding
        # label mapping to WTE-legend
        if args.source_name is not None:
            LOGGER.info('Source labels: {}'.format(args.source_name))
            y_pred = array_replace(y_pred,
                                   LC_LOOKUPS[args.source_name].to_numpy())

        # check if target dataset name is specified and apply corresponding
        # label mapping to WTE-legend
        if args.target_name is not None:
            LOGGER.info('Target labels: {}'.format(args.target_name))
            y_true = array_replace(y_true,
                                   LC_LOOKUPS[args.target_name].to_numpy())

        # exclude missing values
        mask = ((y_true != WteLandCover.NoData.id) &
                (y_pred != WteLandCover.NoData.id))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # filename for output graphics
        fname = '_'.join([args.source.stem, args.target.stem])

        # calculate confusion matrix: assume that the target dataset represents
        # the true state of the land cover
        LOGGER.info('Computing confusion matrix ...')
        cm = confusion_matrix(y_true, y_pred, labels=list(LABELS.keys()))

        # plot confusion matrix
        fig, _ = plot_confusion_matrix(cm, CLASS_LABELS, cmap='Greens')
        cm_name = GRAPHICS_PATH.joinpath(fname + '_cm.png')
        LOGGER.info('Confusion matrix: {}'.format(cm_name))
        fig.savefig(cm_name, dpi=300, bbox_inches='tight')

        # compute classification report
        LOGGER.info('Computing classification metrics ...')
        report = area_adjusted_classification_report(
            y_true, y_pred, y_pred, target_names=CLASS_LABELS,
            labels=list(LABELS.keys()))

        # plot and save classification report
        fig = plot_classification_report(report, CLASS_LABELS, cmap='Greens')
        report_name = GRAPHICS_PATH.joinpath(fname + '_crep.png')
        LOGGER.info('Classification report: {}'.format(report_name))
        fig.savefig(report_name, bbox_inches='tight', dpi=300)

    else:
        LOGGER.info('{} does not exist.'.format(
            args.source if not args.source.exists() else args.target))
        sys.exit()
