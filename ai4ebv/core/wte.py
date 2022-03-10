"""Combine temperature, moisture, landforms, and land cover to the WTE."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import pathlib

# externals
import pandas as pd
import numpy as np
from matplotlib.colors import to_rgb

# locals
from ai4ebv.core.landcover import WteLandCover
from ai4ebv.core.landforms import SayreLandforms
from pysegcnn.core.utils import img2np, array_replace

# World terrestrial ecosystem components
COMPONENTS = ['temperature', 'moisture', 'landform', 'landcover']

# path to this file
HERE = pathlib.Path(__file__).parent

# path to the repository root
ROOT = HERE.parent.parent

# value to assign to water surfaces in the final WTE layers
WATER = 500

# value to assign to NoData in the final WTE layers
NODATA = 0


def world_terrestrial_ecosystems(temperature, moisture, landforms, landcover):

    """Compute the World Terrestrial Ecosystem identifiers.

    Parameters
    ----------
    temperature : `str` or :py:class:`pathlib.Path` or :py:class`numpy.ndarray`
        The World Temperature domains after `Sayre et al. (2020)`_.
    moisture : `str` or :py:class:`pathlib.Path` or :py:class`numpy.ndarray`
        The World Moisture domains after `Sayre et al. (2020)`_.
    landforms : `str` or :py:class:`pathlib.Path` or :py:class`numpy.ndarray`
        The World Landform domains after `Sayre et al. (2020)`_.
    landcover : `str` or :py:class:`pathlib.Path` or :py:class`numpy.ndarray`
        The World Landcover domains after `Sayre et al. (2020)`_.

    Returns
    -------
    code : :py:class`numpy.ndarray`
        The code as constructed using the different components.
    wte : :py:class`numpy.ndarray`
        The corresponding World Terrestrial Ecosystem identifiers.

    .. _Sayre et al. (2020):
        https://doi.org/10.1016/j.gecco.2019.e00860

    """
    # read input layers to numpy arrays
    layers = {k: img2np(v) for k, v in zip(COMPONENTS, [temperature, moisture,
                                                        landforms, landcover])}

    # generate WTE-code: 1000 * temp + 100 * moist + 10 * landform + landcover
    code = (1000 * layers['temperature'] + 100 * layers['moisture'] +
            10 * layers['landform'] + layers['landcover'])

    # code for missing values (LC=255 or LF=255)
    missing = np.where((layers['landcover'] == WteLandCover.NoData.id) |
                       (layers['landform'] == SayreLandforms.NoData.id))
    code[missing] = NODATA

    # read the World Terrestrial Ecosystem table and create lookup for the code
    # to the ecosystem ID
    table = pd.read_csv(ROOT.joinpath(
        'Data','Realm_WE_attribute_Table_DraftSort_colors.csv'))
    lookup = table[['Code', 'WE_ID']].to_numpy()

    # replace code by World Terrestrial Ecosystem identifier using the lookup
    # table
    wte = array_replace(code, lookup)

    # check if there are some undefined ecosystem identifiers and replace by
    # NoData
    not_defined = ~np.isin(wte, lookup[:, 1])
    wte[not_defined] = NODATA

    # code for surface water (LC=7) is mapped to constant WATER
    code[np.where(layers['landcover'] == WteLandCover.Water.id)] = WATER
    wte[np.where(layers['landcover'] == WteLandCover.Water.id)] = WATER

    return code, wte


def wte_colormap(filename):
    """Create the colormap for the World Terrestrial Ecosystems.

    Parameters
    ----------
    filename : `str` or :py:class:`pathlib.Path`
        File to save the colormap.

    Returns
    -------
    colormap : :py:class:`pandas.DataFrame`
        The World Terrestrial Ecosystem colormap.

    """
    # read the World Terrestrial Ecosystem table
    df = pd.read_csv(ROOT.joinpath(
        'Data','Realm_WE_attribute_Table_DraftSort_colors.csv'))

    # get columns for colormap and remove duplicates
    WTEs = df[['WE_ID', 'WEcosystm', 'WTE_HEX']].sort_values(
        'WE_ID').drop_duplicates()

    # rename columns
    WTEs.rename(columns={'WE_ID': 'Ecosystem id',
                         'WEcosystm': 'Ecosystem name'}, inplace=True)

    # convert hex color code to rgb
    colors = pd.DataFrame(columns=['Red', 'Green', 'Blue'])
    for idx, color in WTEs.WTE_HEX.iteritems():
        color = pd.DataFrame([[int(c * 255) for c in to_rgb(color)]],
                             columns=['Red', 'Green', 'Blue'], index=[idx])
        colors = colors.append(color)

    # create colormap: drop hex color code
    colormap = pd.concat(
        [WTEs['Ecosystem id'], colors, WTEs['Ecosystem name']], axis=1)

    # add row for surface water
    colormap = colormap.append(pd.DataFrame(
        [[WATER, *list(WteLandCover.Water.color), 'Surface water']],
        columns=colormap.columns))

    # save colormap to file
    colormap.to_csv(filename, sep=',', index=False)

    return colormap
