"""Classification configuration file."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import pathlib

# locals
#from ai4ebv.core.legend import SUPPORTED_LC_PRODUCTS
from ai4ebv.core.constants import (ALPS_TILES, HIMALAYAS_TILES, LISS_TILES,
                                   STT_TILES, HIMALAYAS_SITE, SMALL_TILES)

# path to this file
HERE = pathlib.Path(__file__).parent

# whether running in the Azure cloud or on local cluster
# AZURE = False
AZURE = True

# whether to mosaic model predictions for multiple tiles
MOSAIC = True
# MOSAIC = False

# year of interest
YEAR = 2017

# months of relevant HLS image data
MONTHS = [6, 7, 8]
# MONTHS = []

# HLS tiles of interest
# TILES = LISS_TILES
# TILES = SMALL_TILES
TILES = ['32TPS']
# TILES = ALPS_TILES
# TILES = HIMALAYAS_SITE
TILES = tuple([tile.lstrip('T') for tile in TILES])
#assert (set(TILES).intersection(ALPS_TILES) or
#        set(TILES).intersection(HIMALAYAS_TILES))

# -----------------------------------------------------------------------------
# Training data sampling ------------------------------------------------------
# -----------------------------------------------------------------------------

# whether to overwrite existing training data
OVERWRITE_TRAINING_DATA = False

# number of pixels per class to select for training
NPIXEL = 1000

# use spectral indices as classification features
# USE_INDICES = False
USE_INDICES = True

# number of pixels defining radius around a training pixel, within which no
# other training pixel should be sampled
# NOTE: BUFFER=None or BUFFER=0 means pixels are just selected randomly
#       BUFFER>=1 means pixels are selected based on the buffer size
# WARNING: A large buffer size in combination with a large pixels size (NPIXEL)
#          can result in significantly longer sampling times
BUFFER = 3

# percent of spatial coverage, below which to drop a scene from the training
# data generation
DROPNA = 80

# percent of cloud coverage, above which to drop a scene from the training data
# generation
DROPQA = 50

# land cover labels to use
LC_LABELS = 'LISS'
#assert LC_LABELS in SUPPORTED_LC_PRODUCTS

# -----------------------------------------------------------------------------
# Classification --------------------------------------------------------------
# -----------------------------------------------------------------------------

# spatial chunk size for multispectral time series during model training
# this parameter is used by Dask to parallelize computations
TILE_SIZE = (256, 256)

# overwrite existing hls time series objects
OVERWRITE_TIME_SERIES = False

# whether to apply quality assessment layer on time series
APPLY_QA = False
# APPLY_QA = True

# whether to predict using classification features or raw time series
FEATURES = True
# FEATURES = False

# whether to compute classification features on annual or seasonal scale
SEASONAL = False
# SEASONAL = True

# whether to use digital elevation model as additional input layer
DEM = False
# DEM = True
