"""Train a neural network for each tile."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import re
import sys
import time
import logging
from logging.config import dictConfig
from datetime import timedelta

# externals
import numpy as np
import xarray as xr
import torch.optim as optim
from osgeo import gdal
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# locals
from ai4ebv.core.dataset import HLSDataset, TabularDataset
from ai4ebv.core.sample import TrainingDataFactory
from ai4ebv.core.constants import (LABELS, OLABELS, ALPS_TILES, STT_TILES,
                                   TILE_PATTERN, HIMALAYAS_SITE, ALPS_CRS,
                                   HIMALAYAS_CRS)
from ai4ebv.core.models import FCN
from ai4ebv.core.predict import predict_hls_tile
from ai4ebv.core.legend import Legend, Legend2Wte
from ai4ebv.core.landcover import WteLandCover
from ai4ebv.core.utils import mosaic_tiles
from ai4ebv.main.io import (ROOT, TS_PATH, MODEL_PATH, CLASS_PATH, TRAIN_PATH,
                            LC_LAYERS, DEM_LAYERS)
from ai4ebv.main.config import (TILES, YEAR, MONTHS, NPIXEL, BUFFER, DROPNA,
                                DROPQA, OVERWRITE_TIME_SERIES, LC_LABELS,
                                AZURE, USE_INDICES, DEM, APPLY_QA, MOSAIC,
                                OVERWRITE_TRAINING_DATA, FEATURES, SEASONAL)
from ai4ebv.main.torch_config import (MODEL, BATCH_SIZE, TORCH_OVERWRITE, LR,
                                      TRAIN_CONFIG, TILE_SIZE)
from pysegcnn.core.logging import log_conf
from pysegcnn.core.trainer import NetworkTrainer, LogConfig
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
    # CHECK MODEL CONFIGURATION -----------------------------------------------
    # -------------------------------------------------------------------------
    if MODEL is FCN and FEATURES:
        LOGGER.info('Cannot apply CNN on classification features.')
        sys.exit()

    # -------------------------------------------------------------------------
    # INITIALIZE LOOP OVER TILES ----------------------------------------------
    # -------------------------------------------------------------------------

    # list of output files: required to generate mosaic
    y_layers = []
    p_layers = []
    q_layers = []
    for tile in TILES:
        LogConfig.init_log('Processing tile {}.'.format(tile))

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
        # LAND COVER LABELS ---------------------------------------------------
        # ---------------------------------------------------------------------

        # get the land cover labels for the current tile
        lc = LC_LAYERS[LC_LABELS][tile]
        ds = gdal.Open(str(lc))

        # ---------------------------------------------------------------------
        # MODEL CONFIGURATION -------------------------------------------------
        # ---------------------------------------------------------------------

        # model output path
        mpath = MODEL_PATH.joinpath(tile).joinpath(str(YEAR))
        if not mpath.exists():
            mpath.mkdir(parents=True, exist_ok=True)

        # generate a file to save model configuration
        state_file = HLSDataset.state_file(
            model=MODEL.__name__, labels=LC_LABELS, npixel=NPIXEL,
            features=FEATURES, use_indices=USE_INDICES, dem=DEM, qa=APPLY_QA,
            mode='single', months=MONTHS, year=YEAR, seasonal=SEASONAL)
        state_file = mpath.joinpath(state_file)
        LOGGER.info('Initializing model state file: {}'.format(state_file))

        # check if output paths exists
        tpath = CLASS_PATH.joinpath('v1.0', tile, str(YEAR))
        dpath = TRAIN_PATH.joinpath('v1.0', tile, str(YEAR))
        for path in [tpath, dpath]:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

        # generate filenames
        fname = tpath.joinpath(state_file.stem + '_{}.tif'.format(tile))
        pname = tpath.joinpath(fname.name.replace('.tif', '_prob.tif'))
        qname = dpath.joinpath(HLSDataset.time_series(tile, str(YEAR), MONTHS).
                               replace('.nc',
                                       '_'.join(['', LC_LABELS, 'qa.tif'])))

        # check if output layers exist and whether to overwrite
        if fname.exists() and not TORCH_OVERWRITE:
            # store predictions and associated probabilities to generate mosaic
            LOGGER.info('{} already exists. Aborting classification ...'
                        .format(fname))
            y_layers.append(fname)
            p_layers.append(pname)
            q_layers.append(qname)
            continue

        # initialize model training
        else:
            # -----------------------------------------------------------------
            # SAMPLE TRAINING DATA --------------------------------------------
            # -----------------------------------------------------------------

            # generate filenames: training data
            sname = dpath.joinpath(time_series.name.replace(
                time_series.suffix, '_'.join(['', LC_LABELS, 'samples.tif'])))
            tname = dpath.joinpath(time_series.name.replace(
                time_series.suffix, '_'.join(['', LC_LABELS, 'train.tif'])))
            dname = dpath.joinpath(time_series.name.replace(
                time_series.suffix, '_'.join(['', LC_LABELS, 'train.nc'])))
            qname = dpath.joinpath(time_series.name.replace(
                time_series.suffix, '_'.join(['', LC_LABELS, 'qa.tif'])))

            # check if training data exists
            if dname.exists() and not OVERWRITE_TRAINING_DATA:
                LogConfig.init_log('Existing training data: {}'.format(dname))
                training_data = xr.open_dataset(dname)
            else:
                # instanciate training data factory
                factory = TrainingDataFactory(hls_ts, lc, LC_LABELS,
                                              qa=APPLY_QA)

                # generate training and validation data
                training_data, samples = factory.generate_training_data(
                    factory.hls_ts, factory.samples, factory.class_labels,
                    NPIXEL, buffer=BUFFER)

                # save training data spectra: NetCDF file
                LogConfig.init_log('Saving training data.')
                LOGGER.info('Saving: {}'.format(dname))
                training_data.to_netcdf(dname, engine='h5netcdf')

                # save sampling and training data layer
                np2tif(factory.samples, filename=sname, overwrite=True,
                       no_data=Legend.NoData.id, src_ds=ds)
                np2tif(samples, filename=tname, no_data=Legend.NoData.id,
                       src_ds=ds, overwrite=True)

                # save pixel quality assessment layer: proportion of valid time
                # steps
                np2tif(factory.qa, filename=qname, no_data=np.nan, src_ds=ds,
                       overwrite=True)

            # load training data to memory
            inputs, labels = TrainingDataFactory.load_training_data(
                training_data, features=FEATURES, use_indices=USE_INDICES)

            # -----------------------------------------------------------------
            # DIGITAL ELEVATION MODEL -----------------------------------------
            # -----------------------------------------------------------------

            # get the digital elevation model
            dem = DEM_LAYERS[tile] if DEM else None
            if dem is not None:
                # dem features: elevation, slope and aspect
                dem_features = TrainingDataFactory.dem_features(
                    dem, add_coord={'time': training_data.time} if not FEATURES
                    else None)

                # subset to training data
                dem_features = dem_features.sel(
                    y=training_data.y, x=training_data.x).to_array().values
                dem_features = dem_features.swapaxes(0, -1).swapaxes(1, -1)
                inputs = np.concatenate((inputs, dem_features), axis=1)

            # -----------------------------------------------------------------
            # INSTANCIATE TRAINING AND VALIDATION DATASET ---------------------
            # -----------------------------------------------------------------

            # split sampled pixels into two parts: training and validation set
            # training and validation datasets are stratified by class
            (train_inputs, valid_inputs, train_labels, valid_labels) = (
                train_test_split(inputs, labels, test_size=0.2,
                                 stratify=labels))

            # multivariate time series training and validation dataset
            train_ds = TabularDataset(train_inputs, train_labels)
            valid_ds = TabularDataset(valid_inputs, valid_labels)

            # create training and validation dataloaders
            train_dl = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
            valid_dl = DataLoader(
                valid_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

            # -----------------------------------------------------------------
            # INSTANCIATE THE MODEL -------------------------------------------
            # -----------------------------------------------------------------

            # network architecture
            if FEATURES:
                net = MODEL(inputs.shape[-1], len(LABELS), state_file)
            else:
                net = MODEL(inputs.shape[1], len(LABELS), state_file)

            # optimizer
            optimizer = optim.Adam(net.parameters(), lr=LR)

            # -----------------------------------------------------------------
            # TRAIN MODEL -----------------------------------------------------
            # -----------------------------------------------------------------

            # instanciate the training class
            trainer = NetworkTrainer(
                net, optimizer, net.state_file, train_dl, valid_dl,
                **TRAIN_CONFIG)

            # train network
            model_state = trainer.train()

        # ---------------------------------------------------------------------
        # INITIALIZE PREDICTION -----------------------------------------------
        # ---------------------------------------------------------------------
        LogConfig.init_log('Predicting tile {}.'.format(tile))

        # model predictions
        y_pred, y_prob = predict_hls_tile(
            hls_ts, net, tile_size=TILE_SIZE, features=FEATURES,
            use_indices=USE_INDICES, dem=dem)

        # replace model labels with original labels
        y_pred = HLSDataset.transform_gt(y_pred, original_labels=OLABELS,
                                         invert=True).astype(np.int16)

        # map working legend to WTE-land cover legend
        y_pred_wte = array_replace(y_pred, Legend2Wte.to_numpy())
        np2tif(y_pred_wte, no_data=WteLandCover.NoData.id, overwrite=True,
               filename=str(fname).replace(fname.suffix, '_wte.tif'),
               src_ds=ds)

        # ---------------------------------------------------------------------
        # SAVE PREDICTIONS AS GEOTIFF -----------------------------------------
        # ---------------------------------------------------------------------

        # save predictions and associated probabilities as GeoTiff
        np2tif(y_pred, filename=fname, no_data=Legend.NoData.id,
               src_ds=ds, overwrite=True)
        np2tif(y_prob, filename=pname, no_data=np.nan, src_ds=ds,
               overwrite=True)

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
