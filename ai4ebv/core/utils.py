"""Utility functions."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import re
import itertools
import logging
import warnings
import pathlib
import tarfile
import rasterio
from rasterio import windows
from rasterio.merge import merge
from rasterio.transform import Affine
from joblib import Parallel, delayed

# externals
import numpy as np
from scipy.signal import medfilt, find_peaks
import matplotlib.pyplot as plt

# locals
from ai4ebv.core.landcover import WteLandCover
from ai4ebv.core.constants import HLS_TILE
from pysegcnn.core.utils import (is_divisible, img2np, reconstruct_scene,
                                 reproject_raster)

# module level logger
LOGGER = logging.getLogger(__name__)


def normalized_difference(b1, b2):
    """Compute the normalized difference between two spectral bands.

    Parameters
    ----------
    b1 : :py:class:`numpy.ndarray`
        First spectral band.
    b2 : :py:class:`numpy.ndarray`
        Second specral band.

    Returns
    -------
    nd : :py:class:`numpy.ndarray`
        Normalized difference between ``b1`` and ``b2``.

    """
    # clip values in the range of [-1, 1]
    with warnings.catch_warnings():
        # catch RunTimeWarning associated with zero divisions
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        return ((b1 - b2) / (b1 + b2)).clip(-1, 1)


def integrated_forest_index(scene, bands, ref=None, window_size=300,
                            ndvi_thres=0.2, nbins=50, medfilt_size=3,
                            mean_filter=True, ifi=True):
    """Compute the integrated forest index (IFI) after `Huang et al. (2008)`_.

    Parameters
    ----------
    scene : :py:class:`xarray.Dataset`
        The multispectral scene to compute the IFI for.
    bands : `list` [`str`], optional
        The spectral bands to use to compute the IFI.
    ref : :py:class:`numpy.ndarray`, optional
        Reference forest mask. The default is `None`.
    window_size : `int`, optional
        Size of the moving window in pixels. The default is `300`.
    ndvi_thres : `float`, optional
        Threshold of the normalized difference vegetation index below which to
        mask dark objects. The default is `0.2`.
    nbins : `int`, optional
        Number of window histogram bins when determining forest peak. The
        default is `50`.
    medfilt_size : `int`, optional
        Size of the median filter applied to the histograms in bins. The
        default is `3`.
    mean_filter : `bool`, optional
        Whether to apply the median filter. The default is `True`.
    ifi : `bool`, optional
        Whether to return the IFI or the forest mask as delineated by the
        forest peak detection only. The default is `True`.

    Returns
    -------
    forest : :py:class:`xarray.DataArray`
        The forest mask as delineated by the forest peak detection
        (``ifi=False``) or the integrated forest index (``ifi=True``).

    .. _Huang et al. (2008):
        https://www.sciencedirect.com/science/article/pii/S0034425707003951

    """
    # helper function
    def forest_peak(w):

        # check whether any values of the current local window are valid
        if np.isnan(w).all():
            # skip windows with no valid values
            return

        # generate the histogram bins
        bins = np.linspace(np.nanmin(w), np.nanmax(w), nbins)

        # compute the histogram of the local window
        n, bins = np.histogram(w, bins=bins)

        # apply median filter
        n_med, bins_med = (medfilt(n, medfilt_size),
                           medfilt(bins, medfilt_size))

        # identify the forest peak: first maximum of the frequency values

        # check whether the peak occurs at the beginning of the histogram
        if np.argmax(n_med) == 0:
            # if the peak occurs within the first bin,
            # scipy.signal.find_peaks fails, since the bin values decrease
            # afterwards
            peaks = np.array([1])
        else:
            # if the peak occurs after the first bin,
            # scipy.signal.find_peaks correctly extracts the peak values
            peaks, _ = find_peaks(n_med)

        # define pair of thresholds to identify forest pixels
        thresholds = (bins_med[0], bins_med[peaks[0]])

        # forest mask
        forest = (w >= thresholds[0]) & (w <= thresholds[1])

        # forest peak
        peak = thresholds[1]

        return forest, peak

    # -------------------------------------------------------------------------
    # MASK DARK OBJECTS -------------------------------------------------------
    # -------------------------------------------------------------------------

    # create a copy of the input scene
    image = scene.copy()

    # calculate the normalized difference vegetation index
    # mask dark objects based on a threshold on the ndvi
    LOGGER.debug('Masking dark objects with NDVI <= {:.2f}'.format(ndvi_thres))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        # image = image.where(image.ndvi >= ndvi_thres, other=np.nan)

    # drop unnecessary variables
    image = image.drop_vars([var for var in scene.data_vars if var not in
                             bands or var == 'qa'])

    # -------------------------------------------------------------------------
    # CREATE LOCAL WINDOWS ----------------------------------------------------
    # -------------------------------------------------------------------------

    # the dimensionality of the input scene
    height, width = len(image.y), len(image.x)

    # get the visible red band to compute local image histograms
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        red = image.red.values

    # divide the scene into local windows of size window_size
    nwin, padding = is_divisible((height, width), window_size, pad=True)
    windows = img2np(red, tile_size=window_size, pad=True, cval=np.nan)

    # check whether a reference forest mask is specified
    if ref is not None:
        ref = img2np(ref, tile_size=window_size, pad=True, cval=0)

    # -------------------------------------------------------------------------
    # COMPUTE FOREST PEAKS ----------------------------------------------------
    # -------------------------------------------------------------------------

    # iterate over the local windows
    LOGGER.debug('Computing local image histograms ...')

    # determine forest pixels within current local window
    forest_windows = Parallel(n_jobs=-1)(
        delayed(forest_peak)(w) for w in windows[:, ...])

    # forest mask and associated peak thresholds
    forest = np.asarray([w[0] for w in forest_windows])
    forest_peaks = np.asarray([w[1] for w in forest_windows])

    # consistency check: compare delineated forest pixels with reference
    # forest mask
    if ref is not None:
        LOGGER.debug('Applying constistency check: reference forest mask ...')

        # percent of delineated forest pixels within each local window
        npixel = np.count_nonzero(forest, axis=(-1, -2)) / (window_size ** 2)

        # percent of reference forest pixels within each local window
        npixel_ref = np.count_nonzero(ref, axis=(-1, -2)) / (window_size ** 2)

        # check where the percentage of delineated forest pixels exceeds the
        # percentage of reference forest pixels
        discard = npixel > np.repeat(np.expand_dims(npixel_ref, axis=1),
                                     npixel.shape[1], axis=1)

        # discard local windows exceeding reference forest mask
        forest[np.where(discard)] = 0

    # consistency check: compare local window forest peaks to mean forest peak
    # value of the scene
    if mean_filter:
        LOGGER.debug('Applying constistency check: forest peak statisics ...')
        discard = forest_peaks > (forest_peaks.mean() + forest_peaks.std())
        forest[np.where(discard), ...] = np.zeros(shape=2 * (window_size, ))

    # reshape local windows to original shape
    forest = reconstruct_scene(forest)

    # clip mask extent to scene extent
    forest = forest[...,
                    padding[2]:forest.shape[-2] - padding[0],
                    padding[1]:forest.shape[-1] - padding[3]]

    # -------------------------------------------------------------------------
    # COMPUTE INTEGRATED FOREST INDEX -----------------------------------------
    # -------------------------------------------------------------------------
    if ifi:
        # get the spectra of the delineated forest training pixels
        forest_spectra = image.where(forest.astype(bool), other=np.nan)

        # spectral mean and standard deviation of forest training pixels
        f_mean = forest_spectra.mean(dim=('y', 'x'), skipna=True)
        f_stdv = forest_spectra.std(dim=('y', 'x'), skipna=True)

        # compute integrated forest index
        LOGGER.debug('Computing integrated forest index ...')

        # compute distance to spectral centroid of forest in each band
        distance = ((scene.drop_vars([var for var in scene.data_vars if var not
                                      in bands or var == 'qa']) - f_mean) /
                    f_stdv) ** 2

        # sum distances over all spectral bands
        forest = np.sqrt((distance.to_array().sum(axis=0, skipna=False) /
                          len(image.data_vars)))

    return forest


def extract_hls_tiles(source, target, tiles=None, year=None, overwrite=False):
    """Extract HLS tiles from a tar archive.

    Parameters
    ----------
    source : :py:class:`pathlib.Path` or `str`
        Path to the archive to extract.
    target : :py:class:`pathlib.Path` or `str`
        Path to save extracted files.
    tiles : `list` [`str`] or `None`, optional
        Name of the Sentinel-2 tiles of interest. If ``tile=None``, all tiles
        are extracted. The default is `None`.
    year : `int` or `None`, optional
        Year of interest. If ``year=None``, all years are extracted. The
        default is `None`.
    overwrite : `bool`, optional
        Whether to overwrite extracted files. The default is `False`.

    """
    # open tar archive
    with tarfile.open(str(source), mode='r') as tar:
        # get the directories/files in the archive
        members = tar.getmembers()

        # search for files of the specified tiles only
        if tiles is not None:
            # strip leading T in tiles, if specified
            tiles = [t.lstrip('T') for t in tiles]

            # files of the specified tiles
            members = [m for m in members if re.search(HLS_TILE, m.path) and
                       re.search(HLS_TILE, m.path)[0].lstrip('T') in tiles]

        # check if year is in path
        if year is not None:
            members = [m for m in members if re.search(str(year), m.path)]

        # iterate over the specified tiles
        if tiles is not None and year is not None:
            for tile in tiles:
                # files of the specified tile
                tile_members = [m for m in members if re.search(tile, m.path)]

                # change relative output paths to tile/year/filename
                for m in tile_members:
                    # original filepath
                    source_path = pathlib.Path(m.path)

                    # sensor: L30 or S30
                    sensor = source_path.name.split('.')[1]

                    # target filepath: tile/year/sensor/filename
                    target_path = pathlib.Path(tile).joinpath(
                        str(year), sensor)
                    target_path = str(target_path.joinpath(source_path.name))

                    # overwrite original with target filepath
                    m.path = target_path

                # sort members to extract by filename
                tile_members = sorted(tile_members, key=lambda x: x.path)

                # extract files from archive
                for m in tile_members:
                    # check if file is already extracted at target path
                    target_path = target.joinpath(m.path)
                    if target_path.exists() and not overwrite:
                        LOGGER.info('{} exists.'.format(target_path))
                    else:
                        # extract file
                        LOGGER.info('extract {}'.format(target_path))
                        tar.extract(m, path=str(target))


def spatial_class_difference(y_true, y_pred):
    """Compute class-wise spatial difference.

    Parameters
    ----------
    y_true : :py:class:`numpy.ndarray`
        Ground truth.
    y_pred : :py:class:`numpy.ndarray`
        Model prediction.

    Returns
    -------
    differences : :py:class:`numpy.ndarray`
        Class-wise labeled difference.

    """
    # initialize array highlighting spatial class-wise differences
    differences = (np.ones(shape=y_true.shape, dtype=np.int16) *
                   WteLandCover.NoData.id)

    # iterate over the labels
    nperms = 0
    for label_true in WteLandCover:
        # skip NoData label
        if label_true.id == WteLandCover.NoData.id:
            continue

        # check the pixels of the current label in the reference
        mask_true = y_true == label_true.id

        # iterate over the labels
        for label_pred in WteLandCover:
            # skip NoData label
            if label_pred.id == WteLandCover.NoData.id:
                continue

            # skip correctly classified pixels
            if label_true.id == label_pred.id:
                continue

            # check the pixels of the current label in the estimator
            mask_pred = y_pred == label_pred.id

            # write pixels matching conditions to array
            differences[np.where(mask_true & mask_pred)] = nperms
            nperms += 1

    return differences


def class_difference_colormap(labels, fname=None, cmap='rainbow'):
    """Generate a colormap for spatial class-wise difference visualization.

    Parameters
    ----------
    labels : :py:class:`enum.EnumMeta`
        Class lables.
    fname : `str` or :py:class:`pathlib.Path`, optional
        Filename to save colormap. The default is `None`.
    cmap : `str`, optional
        A colormap supported by :py:func:`matplotlib.pyplot.get_cmap`. The
        default is 'rainbow'.

    Returns
    -------
    cm : :py:class:`matplotlib.colors.LinearSegmentedColormap`
        The matplotlib colormap.

    """
    # iterate over all possible permutations of the labels
    permutations = {}
    nperms = 0
    for perm in itertools.product(labels, labels):
        # skip the NoData class
        if perm[0].id == labels.NoData.id or perm[1].id == labels.NoData.id:
            continue
        # skip permutation of label to itself
        elif perm[0].id == perm[1].id:
            continue
        else:
            permutations[nperms] = ('{} ({})'.format(
                ' as '.join([perm[0].name, perm[1].name]), nperms))
            nperms += 1

    # define the colormap
    cm = plt.get_cmap(cmap, nperms)
    for k, v in permutations.items():
        permutations[k] = (*[scaled * 255 for scaled in cm(k)], ) + (v,)

    # write colormap to file, if specified
    if fname is not None:
        fname = pathlib.Path(fname)
        fname = str(fname).replace(fname.suffix, '.clr')

        # name of columns
        cols = ['ID', 'R', 'G', 'B', 'A', 'DESCRIPTION']

        # rows to write to file
        rows = [' '.join([str(k), *[str(val) for val in v]]) for k, v in
                permutations.items()]
        with open(str(fname), 'w') as file:
            file.write('{}\n'.format(' '.join(cols)))
            [file.write('{}\n'.format(row)) for row in rows]

    return cm


def mosaic_tiles(layers, layers_prob=[], targets=None, trg_crs=3035,
                 resolution=(30, 30), no_data=WteLandCover.NoData.id,
                 **kwargs):
    """Mosaic model predictions based on a-posteriori probabilities.

    Inspired and guided by :py:func:`rasterio.merge.py`, see `here`_.

    .. _here:
        https://github.com/mapbox/rasterio/blob/master/rasterio/merge.py

    Parameters
    ----------
    layers : `list` [:py:class:`pathlib.Path` or `str`]
        Tile-wise model predictions.
    layers_prob : `list` [:py:class:`pathlib.Path` or `str`], optional
        Tile-wise a-posteriori probabilities. If not specified, tiles ar merged
        using :py:func:`rasterio.merge.merge`. The default is `[]`.
    targets : `list` [:py:class:`pathlib.Path` or `str`], optional
        List of two paths, (i) for the model prediction mosaic and (ii) for the
        model a-posteriori probability mosaic. If ``targets=None``, the mosaics
        are not saved to disk. If ``layers_prob=[]``, targets can be a single
        path to save the mosaic. The default is `None`.
    trg_crs : `str`, optional
        The target coordinate reference system of the mosaic as EPSG code.
        The default is `3035` (LAEA Europe).
    resolution : `tuple` [`int`, `int`], optional
        Target spatial resolution (res, res) in units of ``trg_crs``. The
        default is `(30, 30)` meters.
    no_data : `float` or `int`, optional
        The target NoData value for the mosaic. The default is `255`.

    kwargs : `dict´
        Additional keyword arguments passed to :py:func:`rasterio.merge.merge`.

    Returns
    -------
    mosaic : :py:class:`numpy.ndarray`
        Mosaic of the model predictions.
    prob_mosaic : :py:class:`numpy.ndarray`
        Mosaic of the a-posteriori probabilities.

    """
    # calculate the extent of the mosaic
    def _max_extent(layers):
        xs, ys = [], []
        for layer in layers:
            # get spatial extent of layer
            with rasterio.open(layer) as lyr:
                # reproject bounds to target coordinate reference system
                left, bottom, right, top = lyr.bounds

                # store extent and resolution of current layer
                xs.extend([left, right])
                ys.extend([bottom, top])

        # compute maximum extent: (left, bottom, right, top)
        return min(xs), min(ys), max(xs), max(ys)

    # reproject input layers to target coordinate reference system
    layers_tmp, prob_layers_tmp = [], []
    for lyr in layers + layers_prob:
        # temporary target dataset
        lyr = pathlib.Path(lyr)
        trg_lyr = lyr.parent.joinpath(lyr.name.replace(lyr.suffix, '_tmp.tif'))

        # reproject to target coordinate reference system
        reproject_raster(str(lyr), str(trg_lyr), epsg=trg_crs,
                         pixel_size=resolution, no_data=0 if lyr in layers_prob
                         else no_data, overwrite=True)

        # replace source raster with reprojected raster
        (layers_tmp.append(trg_lyr) if lyr in layers else
         prob_layers_tmp.append(trg_lyr))

    # check whether probability layers are specified
    if not layers_prob:
        # merge tiles using rasterio.merge.merge
        merge(layers_tmp, res=resolution, dst_path=targets, **kwargs)

        # remove temporary layers from disk
        LOGGER.info('Removing temporary reprojected layers ...')
        for lyr in layers_tmp + prob_layers_tmp:
            LOGGER.info('rm {}'.format(lyr))
            lyr.unlink()

        return

    # spatial extent of output mosaic
    LOGGER.info('Calculating mosaic extent ...')
    dst_w, dst_s, dst_e, dst_n = _max_extent(layers_tmp)

    # geotransform of output mosaic: resolution is 30 meters
    output_transform = Affine.translation(dst_w, dst_n)
    output_transform *= Affine.scale(*(resolution[0], -resolution[1]))

    # create output profile for layers
    y_profile = rasterio.open(layers_tmp[0]).profile
    p_profile = rasterio.open(prob_layers_tmp[0]).profile

    # number of rows and columns
    ncols, nrows = (int((dst_e - dst_w) / resolution[0]),
                    int((dst_n - dst_s) / resolution[1]))

    # update output profiles
    for profile in [y_profile, p_profile]:
        profile.update({'height': nrows, 'width': ncols,
                        'transform': output_transform, 'count': 1})

    # adjust bounds to fit
    dst_e, dst_s = output_transform * (ncols, nrows)

    # initialize mosaic: fill with NoData value
    mosaic = np.ones((nrows, ncols), dtype=np.int16) * WteLandCover.NoData.id
    prob_mosaic = np.zeros((nrows, ncols), dtype=np.float32)

    # iterate over the classified layers
    for layer, prob in zip(layers_tmp, prob_layers_tmp):
        LOGGER.info('Mosaicking layer: {}'.format(layer))
        # open classification and corresponding probabilities
        y_pred = rasterio.open(layer)
        y_prob = rasterio.open(prob)

        # get extent of current layer
        src_w, src_s, src_e, src_n = y_pred.bounds

        # compute spatial intersection between current layer and mosaic
        int_w = src_w if src_w > dst_w else dst_w
        int_s = src_s if src_s > dst_s else dst_s
        int_e = src_e if src_e < dst_e else dst_e
        int_n = src_n if src_n < dst_n else dst_n

        # compute the source window
        src_window = windows.from_bounds(
            int_w, int_s, int_e, int_n, y_pred.transform)

        # compute the destination window
        dst_window = windows.from_bounds(
            int_w, int_s, int_e, int_n, output_transform)

        # calculate the extent of the current window in the destination window
        src_window = src_window.round_shape(pixel_precision=0)
        dst_window = dst_window.round_shape(pixel_precision=0)
        trows, tcols = dst_window.height, dst_window.width
        temp_shape = (1, trows, tcols)

        # read classifications and probabilities of current window
        labl = y_pred.read(out_shape=temp_shape, window=src_window,
                           boundless=False, masked=True, indexes=1)
        prob = y_prob.read(out_shape=temp_shape, window=src_window,
                           boundless=False, masked=True, indexes=1)

        # retrieve classifications and associated probabilities in the mosaic
        dst_window = dst_window.round_offsets(pixel_precision=0)
        roff, coff = max(0, dst_window.row_off), max(0, dst_window.col_off)
        region = mosaic[roff:roff + trows, coff:coff + tcols]
        prob_region = prob_mosaic[roff:roff + trows, coff:coff + tcols]

        # determine classification with highest probability
        condition = prob.data > prob_region

        # overwrite probabilities
        prob_region[condition] = prob.data[condition]
        prob_mosaic[roff:roff + trows, coff:coff + tcols] = prob_region

        # overwrite classifications
        region[condition] = labl.data[condition]
        mosaic[roff:roff + trows, coff:coff + tcols] = region

    # remove temporary layers from disk
    LOGGER.info('Removing temporary reprojected layers ...')
    for lyr in layers_tmp + prob_layers_tmp:
        LOGGER.info('rm {}'.format(lyr))
        lyr.unlink()

    # save mosaics to disk, if target path is specified
    if targets is not None:
        # write mosaic of model predictions and probabilities to disk
        for target, arr, profile in zip(targets, [mosaic, prob_mosaic],
                                        [y_profile, p_profile]):
            with rasterio.open(target, mode='w', **profile) as trg:
                trg.write(arr, 1)
                LOGGER.info('Created mosaic: {}'.format(target))

    return mosaic, prob_mosaic
