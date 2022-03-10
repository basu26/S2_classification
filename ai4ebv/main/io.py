"""Input/Output configuration file."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import pathlib
# import datetime

# locals
from ai4ebv.main.config import TILES
from ai4ebv.core.constants import ALPS_TILES, LISS_TILES
from pysegcnn.core.utils import search_files

# -----------------------------------------------------------------------------
# Input and output paths ------------------------------------------------------
# -----------------------------------------------------------------------------

# define path to project root directory: inputs and outputs are stored here
# ROOT = pathlib.Path('C:/Eurac/Projects/AI4EBVS/')
ROOTmy = pathlib.Path('/home/btufail@eurac.edu/Documents/AI4EBV/')
ROOT = pathlib.Path('/mnt/CEPH_PROJECTS/AI4EBV')
# ROOT = pathlib.Path('/mnt/largedisk/AI4EBV')
# ROOT = pathlib.Path('/mnt/drive/AI4EBV')

# path to save time series objects
TS_PATH = ROOTmy.joinpath('EO').joinpath('NETCDF')

# path to save hls tiles
HLS_PATH = ROOTmy.joinpath('EO').joinpath('HLS')

# path to save outputs
TARGET_PATH = ROOTmy.joinpath('OUTPUTS')

# path to save selected training data samples
SAMPLE_PATH = TARGET_PATH.joinpath('Samples')

# path to save classified tiles
CLASS_PATH = TARGET_PATH.joinpath('Classified')

# path to save training data
TRAIN_PATH = TARGET_PATH.joinpath('Training')

# path to save graphics
GRAPHICS_PATH = TARGET_PATH.joinpath('Graphics')

# path to save models
MODEL_PATH = TARGET_PATH.joinpath('Models')

# create output paths
for path in [TS_PATH, HLS_PATH, SAMPLE_PATH, CLASS_PATH, GRAPHICS_PATH,
             MODEL_PATH]:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Global land cover layers ----------------------------------------------------
# -----------------------------------------------------------------------------

# ESA CCI Land Cover
ESACCI_LAYERS = {
    tile: search_files(ROOTmy, '^ESACCI-LC(.*){}.tif$'.format(tile)).pop()
    for tile in TILES}

# initialize land cover dictionary
LC_LAYERS = {'ESACCI': ESACCI_LAYERS, 'CORINE': {}, 'HULC': {}, 'LISS':{}}

# -----------------------------------------------------------------------------
# Regional land cover layers ----------------------------------------------------
# -----------------------------------------------------------------------------

# check whether the specified tiles are in the European Alps
if set(TILES).intersection(set(ALPS_TILES)):
    # regional reference land cover layer: Corine (2018)
    CLC_LAYERS = {
        tile: search_files(ROOTmy, '^CORINE(.*){}.tif$'.format(tile)).pop()
        for tile in TILES}
    LC_LAYERS.update({'CORINE': CLC_LAYERS})

    # regional reference land cover layer: Humboldt University (2015)
    HUL_LAYERS = {
        tile: search_files(ROOT, '^europe(.*){}.tif$'.format(tile)).pop()
        for tile in TILES}
    LC_LAYERS.update({'HULC': HUL_LAYERS})

# check whether the specified tiles are covered by the LISS dataset
if set(TILES).intersection(set(LISS_TILES)):

    # local benchmark land cover layer: Liss
    LSS_LAYERS = {
        tile: search_files(ROOT, '^LISS(.*){}.tif$'.format(tile)).pop()
        for tile in LISS_TILES}
    LC_LAYERS.update({'LISS': LSS_LAYERS})

# -----------------------------------------------------------------------------
# Military grid reference system ----------------------------------------------
# -----------------------------------------------------------------------------

# HLS tiling grid
MGRS_GRID = search_files(ROOT, 'S2A_OPER_GIP_TILPAR_MPC(.*).kml$').pop()

# -----------------------------------------------------------------------------
# Digital elevation models ----------------------------------------------------
# -----------------------------------------------------------------------------

# SRTM
DEM_LAYERS = {tile: search_files(ROOT, 'SRTM(.*){}.tif$'.format(tile)).pop()
              for tile in TILES}
