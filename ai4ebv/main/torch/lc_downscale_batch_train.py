"""Train a neural network for all tiles."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys
import time
import logging
from logging.config import dictConfig
from datetime import timedelta

# externals
import numpy as np
import xarray as xr
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from osgeo import gdal

# locals
from ai4ebv.core.dataset import HLSDataset, TabularDataset, PadCollate
from ai4ebv.core.sample import TrainingDataFactory
from ai4ebv.core.models import FCN
from ai4ebv.core.constants import LABELS
from ai4ebv.core.legend import Legend
from ai4ebv.main.io import (ROOT, TS_PATH, MODEL_PATH, CLASS_PATH, TRAIN_PATH,
                            LC_LAYERS, DEM_LAYERS)
from ai4ebv.main.config import (TILES, YEAR, MONTHS, NPIXEL, BUFFER, DROPNA,
                                DROPQA, OVERWRITE_TIME_SERIES, LC_LABELS,
                                AZURE, USE_INDICES, DEM, APPLY_QA,
                                OVERWRITE_TRAINING_DATA, FEATURES, SEASONAL)
from ai4ebv.main.torch_config import (MODEL, BATCH_SIZE, TRAIN_CONFIG,
                                      TORCH_OVERWRITE, LR)
from pysegcnn.core.logging import log_conf
from pysegcnn.core.trainer import NetworkTrainer, LogConfig
from pysegcnn.core.utils import np2tif

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

    # model output path
    mpath = MODEL_PATH.joinpath(str(YEAR))
    if not mpath.exists():
        mpath.mkdir(parents=True, exist_ok=True)

    # generate a file to save model configuration
    state_file = HLSDataset.state_file(
            model=MODEL.__name__, labels=LC_LABELS, npixel=NPIXEL,
            features=FEATURES, use_indices=USE_INDICES, dem=DEM,
            qa=APPLY_QA, months=MONTHS, mode='batch', year=YEAR,
            seasonal=SEASONAL)
    state_file = mpath.joinpath(state_file)
    LOGGER.info('Initializing model state file: {}'.format(state_file))

    # check if the network exists
    if state_file.exists() and not TORCH_OVERWRITE:
         LOGGER.info('Model {} already exists. Aborting training ...'
                     .format(state_file))
         sys.exit()

    # -------------------------------------------------------------------------
    # INITIALIZE LOOP OVER TILES ----------------------------------------------
    # -------------------------------------------------------------------------

    # initialize training data
    train_data = {'inputs': [], 'labels': []}
    for tile in TILES:
        LogConfig.init_log('Processing tile {}.'.format(tile))

        # ---------------------------------------------------------------------
        # OUTPUT FILE NAMES ---------------------------------------------------
        # ---------------------------------------------------------------------

        # check if output paths exists
        tpath = CLASS_PATH.joinpath('v1.0', tile, str(YEAR))
        dpath = TRAIN_PATH.joinpath('v1.0', tile, str(YEAR))
        for path in [tpath, dpath]:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

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
        # SAMPLE TRAINING DATA ------------------------------------------------
        # ---------------------------------------------------------------------

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
            factory = TrainingDataFactory(hls_ts, lc, LC_LABELS, qa=APPLY_QA)

            # generate training and validation data
            training_data, samples = factory.generate_training_data(
                factory.hls_ts, factory.samples, factory.class_labels, NPIXEL,
                buffer=BUFFER)

            # save training data spectra: NetCDF file
            LogConfig.init_log('Saving training data.')
            LOGGER.info('Saving: {}'.format(dname))
            training_data.to_netcdf(dname, engine='h5netcdf')

            # save sampling and training data layer
            np2tif(factory.samples, filename=sname,
                   no_data=Legend.NoData.id, src_ds=ds, overwrite=True)
            np2tif(samples, filename=tname,
                   no_data=Legend.NoData.id, src_ds=ds, overwrite=True)

            # save pixel quality assessment layer: proportion of valid time
            # steps
            np2tif(factory.qa, filename=qname, no_data=np.nan, src_ds=ds,
                   overwrite=True)

        # load training data to memory
        inputs, labels = TrainingDataFactory.load_training_data(
            training_data, features=FEATURES, use_indices=USE_INDICES)

        # ---------------------------------------------------------------------
        # DIGITAL ELEVATION MODEL ---------------------------------------------
        # ---------------------------------------------------------------------

        # get the digital elevation model
        dem = DEM_LAYERS[tile] if DEM else None
        if dem is not None:
            # dem features: elevation, slope and aspect
            dem_features = TrainingDataFactory.dem_features(
                dem, add_coord={'time': training_data.time} if not FEATURES
                else None)

            # subset to training data: (nsamples, nbands, time)
            #                          (nsamples, nfeatures)
            dem_features = dem_features.sel(
                y=training_data.y, x=training_data.x).to_array().values
            dem_features = dem_features.swapaxes(0, -1).swapaxes(1, -1)
            inputs = np.concatenate((inputs, dem_features), axis=1)

        # check if ANN/MLP is used with time series
        if MODEL is not FCN and not FEATURES:
            # reshape inputs to required shape: (nsamples, time, nbands)
            inputs = [ds.swapaxes(1, -1) for ds in inputs]

        # store sampled pixels of current tile and continue
        train_data['inputs'].extend(inputs)
        train_data['labels'].extend(labels)

    # -------------------------------------------------------------------------
    # CREATE TRAINING AND VALIDATION DATASET ----------------------------------
    # -------------------------------------------------------------------------
    inputs = train_data['inputs']
    labels = train_data['labels']

    # split sampled pixels into two parts: training and validation set
    # training and validation datasets are stratified by the class labels
    (train_inputs, valid_inputs, train_labels, valid_labels) = (
        train_test_split(inputs, labels, test_size=0.2, stratify=labels))

    # multivariate time series training and validation dataset
    train_ds = TabularDataset(train_inputs, train_labels)
    valid_ds = TabularDataset(valid_inputs, valid_labels)

    # create training and validation dataloaders
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=False, collate_fn=PadCollate(dim=1))
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=False, collate_fn=PadCollate(dim=1))

    # -------------------------------------------------------------------------
    # INSTANCIATE THE MODEL ---------------------------------------------------
    # -------------------------------------------------------------------------

    # network architecture
    net = MODEL(inputs[0].shape[0], len(LABELS), state_file)

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=LR)

    # -------------------------------------------------------------------------
    # TRAIN MODEL -------------------------------------------------------------
    # -------------------------------------------------------------------------

    # instanciate the training class
    trainer = NetworkTrainer(
        net, optimizer, net.state_file, train_dl, valid_dl, **TRAIN_CONFIG)

    # train network
    model_state = trainer.train()

    # log execution time of script
    LogConfig.init_log('Execution time of script {}: {}'
                       .format(__file__, timedelta(seconds=time.monotonic() -
                                                   start_time)))
