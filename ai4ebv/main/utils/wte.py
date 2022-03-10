"""Build the World Terrestrial Ecosystems."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys
import logging
import pathlib
from logging.config import dictConfig

# externals
import numpy as np
from osgeo import gdal

# locals
from pysegcnn.core.utils import np2tif, search_files
from pysegcnn.core.logging import log_conf
from ai4ebv.core.wte import world_terrestrial_ecosystems, COMPONENTS, NODATA

# module level logger
LOGGER = logging.getLogger(__name__)

# study sites
STUDY_SITES = ['Province', 'Himalayas']

# reference years
REFERENCE_YEARS = np.arange(2015, 2021)

# path to deliverables
DELIVERABLES = pathlib.Path('/mnt/CEPH_PROJECTS/AI4EBV/DELIVERABLES/v1.0')

# base filename for WTE layers
WTE_BASENAME = 'World_Terrestrial_Ecosystems_{}_{}_v1.tif'


if __name__ == '__main__':

    # initialize logging
    dictConfig(log_conf())

    # check whether the input paths exist
    paths = {k: DELIVERABLES.joinpath(k.capitalize()) for k in COMPONENTS}
    for _, path in paths.items():
        if not path.exists():
            LOGGER.info('{} does not exist.'.format(str(path)))
            sys.exit()

    # target path
    wte_path = DELIVERABLES.joinpath('WTE')

    # iterate over the reference years
    for year in REFERENCE_YEARS:
        # iterate over study sites
        for site in STUDY_SITES:
            # paths to search for input files
            input_paths = {k: v.joinpath(site) if k != 'landcover' else
                           v.joinpath(site, str(year)) for k, v in
                           paths.items()}

            # input files
            temperature = search_files(
                input_paths['temperature'], 'worldclim(.*).tiff$').pop()
            moisture = search_files(
                input_paths['moisture'], 'cgiar(.*).tiff$' if
                site == 'Himalayas' else 'chelsa(.*).tiff$').pop()
            landform = search_files(
                input_paths['landform'], 'MERIT(.*)WTE_LF.tif$').pop()
            landcover = search_files(
                input_paths['landcover'], '(.*)wte.tif$').pop()

            # build the World Terrestrial Ecosystem layer from the components
            LOGGER.info('Converting to World Terrestrial Ecosystems ...')
            _, wte = world_terrestrial_ecosystems(
                temperature, moisture, landform, landcover)

            # create target file
            filename = wte_path.joinpath(
                site, WTE_BASENAME.format(site, str(year)))
            if not filename.parent.exists():
                filename.parent.mkdir(parents=True, exist_ok=True)

            # save WTE layer to disk
            np2tif(wte, filename, no_data=NODATA, overwrite=True,
                   src_ds=gdal.Open(str(landcover)), compress=False)
