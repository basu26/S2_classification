"""Compute Hammond landforms from a digital elevation model."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys
import time
import pathlib
import logging
from datetime import timedelta
from joblib import Parallel, delayed
from logging.config import dictConfig

# externals
import numpy as np
from osgeo import gdal

# locals
from ai4ebv.core.landforms import (hammond_landforms, LandformAggregation,
                                   HAMMOND_LANDFORM_CLASSES)
from ai4ebv.main.landforms.config import (SLOPE_NAW, RELIEF_NAW, PROFILE_NAW,
                                          DEM, TARGET, NODATA, OVERWRITE,
                                          NCPUS)
from pysegcnn.core.trainer import LogConfig
from pysegcnn.core.logging import log_conf
from pysegcnn.core.utils import np2tif, array_replace

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # initialize timing
    start_time = time.monotonic()

    # initialize logging
    log_file = TARGET.joinpath('log.txt')
    if log_file.exists():
        log_file.unlink()
    dictConfig(log_conf(log_file))

    # check if a single DEM or a list of DEMs is specified
    LogConfig.init_log('Fetching digital elevation models.')
    DEM = (sorted([pathlib.Path(DEM)]) if isinstance(DEM, str) else
           sorted([pathlib.Path(dem) for dem in DEM]))

    # check if DEMs exist
    for dem in DEM:
        if not dem.exists():
            LOGGER.info('{} does not exist.'.format(dem))
            DEM.remove(dem)

    # check if there still are DEMs to process
    if not any(DEM):
        LogConfig.init_log('None of the specified DEMs exists!')
        sys.exit()

    # output filenames: TARGET/(<Hammond>|<WTE-LF>)/<dem>_(<HLF>|<WTE-LF>).tif
    filenames = [TARGET.joinpath(dem.name.replace(dem.suffix, '_HLF.tif')) for
                 dem in DEM]

    # check if target path exists
    if not TARGET.exists():
        TARGET.mkdir(parents=True, exist_ok=True)
    TARGET.joinpath('Hammond').mkdir(exist_ok=True)
    TARGET.joinpath('WTE-LF').mkdir(exist_ok=True)

    # check if Hammond landform layers exist and whether to overwrite
    for i, f in enumerate(filenames):
        if f.exists() and not OVERWRITE:
            LOGGER.info('{} already exists'.format(f))
            filenames.remove(f)  # remove already processed DEMs from list
        else:
            LOGGER.info('Processing: {}'.format(DEM[i]))

    # log number of DEMs to process
    LogConfig.init_log('Number of DEMs to process: {:d}'.format(len(DEM)))

    # check if there still are DEMs to process
    if not any(filenames):
        LogConfig.init_log('All DEMs are processed!')
        sys.exit()

    # compute hammond landforms in parallel
    LogConfig.init_log('Initializing computation of Hammond landforms.')
    LANDFORMS = Parallel(n_jobs=NCPUS, verbose=51)(
        delayed(hammond_landforms)(dem, SLOPE_NAW, RELIEF_NAW, PROFILE_NAW,
                                   NODATA) for dem in DEM)

    # save landform layers
    LogConfig.init_log('Saving landform layers.')
    for dem, lf, file in zip(DEM, LANDFORMS, filenames):
        # save to GeoTIFF
        np2tif(lf, filename=file.parent.joinpath('Hammond', file.name),
               src_ds=gdal.Open(str(dem)), no_data=NODATA, overwrite=OVERWRITE)

        # aggregate to landforms after Sayre et al. (2020)
        lf[~np.isin(lf, HAMMOND_LANDFORM_CLASSES)] = 0
        lf = array_replace(lf, LandformAggregation.to_numpy())
        lf[lf == 0] = NODATA
        np2tif(lf, filename=file.parent.joinpath('WTE-LF', file.name.replace(
            '_HLF.tif','_WTE-LF.tif')), src_ds=gdal.Open(str(dem)),
            no_data=NODATA, overwrite=OVERWRITE)

    # log execution time of script
    LogConfig.init_log('Execution time of script {}: {}'
                       .format(__file__, timedelta(seconds=time.monotonic() -
                                                   start_time)))
