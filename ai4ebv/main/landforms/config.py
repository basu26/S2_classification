"""Configuration file for computation of Hammond landforms."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import os
import pathlib

# locals
from ai4ebv.core.landforms import RectangularNAW, CircularNAW
from ai4ebv.core.constants import TILE_PATTERN
from pysegcnn.core.utils import search_files

# -----------------------------------------------------------------------------
# Neighborhood analysis windows (NAW) -----------------------------------------
# -----------------------------------------------------------------------------

# NAW for slope, relief, and profile parameters

# ------------------------------------------
# NOTE: do not change the modes of the NAWs!
# ------------------------------------------
SLOPE_NAW = CircularNAW(mode='mean', radius=33)
RELIEF_NAW = CircularNAW(mode='sum', radius=33)
PROFILE_NAW = CircularNAW(mode='sum', radius=33)

# -----------------------------------------------------------------------------
# Input/Output ----------------------------------------------------------------
# -----------------------------------------------------------------------------

# define DEM(s) to process: path or a list of paths
DEM = '/mnt/CEPH_PROJECTS/AI4EBV/INPUTS/DEM/SRTM/SRTM_ALPS.tif'
DEM = search_files('/mnt/CEPH_PROJECTS/AI4EBV/INPUTS/DEM/SRTM/',
                   'SRTM_ALPS' + TILE_PATTERN)

# path to save Hammond landforms
# output filenames: TARGET/(<Hammond>|<WTE-LF>)/<dem>_(<HLF>|<WTE-LF>).tif
TARGET = pathlib.Path('/mnt/CEPH_PROJECTS/AI4EBV/LANDFORMS/Python/')

# whether to overwrite existing Hammond landform layers in TARGET
OVERWRITE = False

# -----------------------------------------------------------------------------
# Computation configuration ---------------------------------------------------
# -----------------------------------------------------------------------------

# NoData value for Hammond landform layers
# ------------------------------------------------------
# NOTE: this needs to be a positive value in [0, 65535],
#       since the landform layers are saved as UInt16
# ------------------------------------------------------
NODATA = 9999

# number of CPU cores to use
NCPUS = -1  # -1 means using all available cores
# NCPUS = os.cpu_count() - 1  # means using all available cores except one
