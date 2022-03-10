"""Apply a pretrained neural network to a HLS time series."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import re
import time
import logging
from logging.config import dictConfig
from datetime import timedelta

# externals
import numpy as np
import xarray as xr
from osgeo import gdal

# locals
from ai4ebv.core.dataset import HLSDataset
from ai4ebv.core.predict import predict_hls_tile
from ai4ebv.core.legend import Legend, Legend2Wte
from ai4ebv.core.landcover import WteLandCover
from ai4ebv.core.constants import OLABELS
from ai4ebv.core.utils import mosaic_tiles
from ai4ebv.core.sample import TrainingDataFactory
from ai4ebv.core.constants import (ALPS_TILES, STT_TILES, HIMALAYAS_SITE,
                                   TILE_PATTERN, ALPS_CRS, HIMALAYAS_CRS)
from ai4ebv.main.io import (ROOT, TS_PATH, MODEL_PATH, CLASS_PATH, TRAIN_PATH,
                            LC_LAYERS, DEM_LAYERS)
from ai4ebv.main.config import (TILES, YEAR, MONTHS, NPIXEL, DROPNA, DROPQA,
                                OVERWRITE_TIME_SERIES, LC_LABELS, AZURE,
                                USE_INDICES, DEM, APPLY_QA, MOSAIC, FEATURES,
                                SEASONAL)
from ai4ebv.main.torch_config import MODEL, TILE_SIZE
from pysegcnn.core.logging import log_conf
from pysegcnn.core.models import Network
from pysegcnn.core.trainer import LogConfig
from pysegcnn.core.utils import np2tif, array_replace

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # initialize timing
    start_time = time.monotonic()

    # -------------------------------------------------------------------------
    # INITIALIZE LOGGING ------------------------------------------------------
    # -------------------------------------------------------------------------
    logfile = ROOT.joinpath('log.txt')

    # remove existing log-files
    if logfile.exists():
        logfile.unlink()

    # initialize logging
    dictConfig(log_conf(logfile))

    # -------------------------------------------------------------------------
    # INSTANCIATE THE MODEL ---------------------------------------------------
    # -------------------------------------------------------------------------

    # model output path
    mpath = MODEL_PATH.joinpath(str(YEAR))

    # model state configuration
    state_file = HLSDataset.state_file(
            model=MODEL.__name__, labels=LC_LABELS, npixel=NPIXEL,
            features=FEATURES, use_indices=USE_INDICES, dem=DEM,
            qa=APPLY_QA, months=MONTHS, mode='batch', year=YEAR,
            seasonal=SEASONAL)
    state_file = mpath.joinpath(state_file)

    # load pretrained model
    net, _ = Network.load_pretrained_model(state_file, MODEL)

    # -------------------------------------------------------------------------
    # INITIALIZE LOOP OVER TILES ----------------------------------------------
    # -------------------------------------------------------------------------

    # list of output files: required to generate mosaic
    y_layers = []
    p_layers = []
    q_layers = []
    for tile in TILES:
        LogConfig.init_log('Predicting tile {}.'.format(tile))

        # ---------------------------------------------------------------------
        # HLS TIME SERIES -----------------------------------------------------
        # ---------------------------------------------------------------------

        # check if time series exists on disk
        time_series = TS_PATH.joinpath(
            tile, str(YEAR), HLSDataset.time_series(tile, str(YEAR), MONTHS))
        if time_series.exists() and not OVERWRITE_TIME_SERIES:
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
                                   overwrite=OVERWRITE_TIME_SERIES)

            # check if at least one valid image is found
            if hls_ts is None:
                continue

        # ---------------------------------------------------------------------
        # DIGITAL ELEVATION MODEL ---------------------------------------------
        # ---------------------------------------------------------------------

        # get the digital elevation model
        dem = DEM_LAYERS[tile] if DEM else None

        # ---------------------------------------------------------------------
        # APPLY MODEL TO TILE -------------------------------------------------
        # ---------------------------------------------------------------------

        # model predictions
        y_pred, y_prob = predict_hls_tile(
            hls_ts, net, tile_size=TILE_SIZE, features=FEATURES,
            use_indices=USE_INDICES, dem=dem)

        # replace model labels with original labels
        y_pred = HLSDataset.transform_gt(y_pred, original_labels=OLABELS,
                                         invert=True).astype(np.int16)

        # ---------------------------------------------------------------------
        # SAVE PREDICTIONS AS GEOTIFF -----------------------------------------
        # ---------------------------------------------------------------------

        # raster dataset defining spatial reference
        ds = gdal.Open(str(LC_LAYERS[LC_LABELS][tile]))

        # check if output paths exists
        tpath = CLASS_PATH.joinpath('v1.0', tile, str(YEAR))
        dpath = TRAIN_PATH.joinpath('v1.0', tile, str(YEAR))
        for path in [tpath, dpath]:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

        # generate filenames
        fname = tpath.joinpath(net.state_file.stem + '_{}.tif'.format(tile))
        pname = tpath.joinpath(fname.name.replace('.tif', '_prob.tif'))
        qname = dpath.joinpath(time_series.name.replace(
            time_series.suffix, '_'.join(['', LC_LABELS, 'qa.tif'])))

        # save predictions and associated probabilities as GeoTiff
        np2tif(y_pred, filename=fname, no_data=Legend.NoData.id,
               src_ds=ds, overwrite=True)
        np2tif(y_prob, filename=pname, no_data=np.nan, src_ds=ds,
               overwrite=True)

        # map working legend to WTE-land cover legend
        y_pred_wte = array_replace(y_pred, Legend2Wte.to_numpy())
        np2tif(y_pred_wte, no_data=WteLandCover.NoData.id, overwrite=True,
               filename=str(fname).replace(fname.suffix, '_wte.tif'),
               src_ds=ds)

        # store predictions and associated probabilities to generate mosaic
        y_layers.append(fname)
        p_layers.append(pname)
        q_layers.append(qname)

    # -------------------------------------------------------------------------
    # GENERATE MOSAICS --------------------------------------------------------
    # -------------------------------------------------------------------------
    if MOSAIC:
        LogConfig.init_log('Initializing mosaicking.')

        # check spatial coverage of mosaic
        if TILES == ALPS_TILES:
            coverage = 'Alps'
        elif TILES == STT_TILES:
            coverage = 'Province'
        elif TILES == HIMALAYAS_SITE:
            coverage = 'Himalayas'
        else:
            coverage = ('Local_Alps' if set(TILES).intersection(ALPS_TILES)
                        else 'Local_Himalaya')

        # target path to save mosaics
        target = CLASS_PATH.joinpath('v1.0', 'Mosaic', coverage, str(YEAR))
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)

        # mosaic filenames
        basename = re.search('.+?(?={})'.format(TILE_PATTERN),
                             str(y_layers[0].name))[0]
        targets = [target.joinpath(basename + '.tif'),
                   target.joinpath(basename + '_prob.tif')]

        # mosaic of classifications and posterior probabilities
        crs = (ALPS_CRS if set(TILES).intersection(ALPS_TILES) else
               HIMALAYAS_CRS)
        mosaic_tiles(y_layers, p_layers, targets=targets, trg_crs=crs)

        # mosaic of classifications in Wte-legend
        wte_mosaic_path = target.joinpath(basename + '_wte.tif')
        wte = TrainingDataFactory.read_labels(targets[0], 'LEGEND')
        np2tif(wte, filename=wte_mosaic_path, no_data=WteLandCover.NoData.id,
               src_ds=gdal.Open(str(targets[0])), overwrite=True)

        # mosaic of pixel quality assessment layer
        mosaic_tiles(q_layers, targets=target.joinpath(basename + '_qa.tif'),
                     method='min', trg_crs=crs)

    # log execution time of script
    LogConfig.init_log('Execution time of script {}: {}'
                       .format(__file__, timedelta(seconds=time.monotonic() -
                                                   start_time)))
