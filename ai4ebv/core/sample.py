"""Training data sampling class."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import logging
import warnings
from joblib import Parallel, delayed

# externals
import numpy as np
import xarray as xr
import dask.array as da
from scipy.signal import convolve2d
from sklearn.preprocessing import minmax_scale
from osgeo import gdal

# locals
from ai4ebv.core.constants import USE_BANDS
from ai4ebv.core.legend import Legend
from ai4ebv.core.utils import normalized_difference
from pysegcnn.core.trainer import LogConfig
from pysegcnn.core.utils import (img2np, array_replace, reconstruct_scene,
                                 tile_topleft_corner)
from ai4ebv.main.config import TILE_SIZE

# module level logger
LOGGER = logging.getLogger(__name__)

# spectral indices to use as classification features
INDEX_FEATURES = ['ndvi', 'ndsi', 'ndbi', 'ndwi']

# classes for which an additional threshold-based filter is applied
SUPERVISED_CLASSES = [Legend.Water_bodies, Legend.Glaciers]


class TrainingDataFactory(object):
    """Automatic training data sampling class."""

    # multispectral bands
    bands = USE_BANDS

    # percentiles for spatio-temporal features
    percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    # class labels
    class_labels = [label for label in Legend if label is not Legend.NoData]

    def __init__(self, hls_ts, labels, label_name, qa=False):
        """Initialize automatic training data sampling class.

        Parameters
        ----------
        hls_ts : :py:class:`xarray.Dataset`
            The HLS time series.
        labels : `str` or :py:class:`pathlib.Path`
            Path to the input land cover product.
        label_name : `str`
            Name of the input land cover product.
        qa : `bool`, optional
            Whether to apply the HLS quality assessment layers. The default is
            `False`.

        """
        # instance attributes
        LogConfig.init_log('Initializing training data factory.')

        # HLS multispectral time series: scale to physical units
        self.hls_ts = self.hls_scale(hls_ts)

        # whether to apply quality assessment layer
        if qa:
            # mask pixels flagged as cloud or shadow by the QA layer
            LOGGER.info('Applying quality assessment layer ...')
            self.hls_ts = self.apply_qa(self.hls_ts)

        # proportion of valid time steps for each pixel
        # self.qa = 1 - self.percent_masked(self.hls_ts)

        # drop quality assessment layers from time series dataset
        # self.hls_ts = self.hls_ts.drop_vars(['qa', 'qbin'])

        # read land cover labels ----------------------------------------------

        # land cover labels
        LOGGER.info('Land cover labels: {}'.format(labels))
        #self.labels = self.read_labels(labels, label_name)
        self.labels = img2np(labels)
        
        # apply spectral filter -----------------------------------------------

        # apply spectral filter to select training data
        self.samples = self.spectral_filter(
            self.hls_ts, self.labels, self.class_labels)

    @staticmethod
    def spectral_filter(hls_ts, labels, class_labels, kernel_size=3, **kwargs):
        """Automatic training data generation algorithm.

        Parameters
        ----------
        hls_ts : :py:class:`xarray.Dataset`
            The HLS time series.
        labels `str` or :py:class:`pathlib.Path`
            Path to the input land cover product.
        class_labels : :py:class:`enum.EnumMeta`
            The land cover classification legend.
        kernel_size : `int`, optional
            Kernel size for the erosion of boundary pixels. The default is `5`.
        **kwargs : `dict`
            Additional keyword arguments passed to :py:meth:`ai4ebv.core.
            sample.TrainingDataFactory.distance_to_centroid`.

        Returns
        -------
        samples : :py:class:`numpy.ndarray`
            The spectrally filtered input land cover product.

        """
        # initialize training data samples array
        LogConfig.init_log('Applying spectral filter to input labels.')
        samples = (np.ones(shape=hls_ts.x.shape + hls_ts.y.shape,
                           dtype=np.int16) * Legend.NoData.id)
        
        def _filter(labels, class_label):
            # erode boundary pixels
            mask = TrainingDataFactory.is_homogeneous(
                labels == class_label.id, kernel_size)
            

            # unsupervised spectral filter: remove outliers
            valid = TrainingDataFactory.distance_to_centroid(
                hls_ts, mask, **kwargs)

            # additional supervised threshold-based spectral filter for:
            #    - Water surfaces: NDWI > 0
            #    - Snow and ice  : NDSI > 0.4 & NIR > 0.11
            if np.sum(valid) == 0:
                return valid
            filtered = TrainingDataFactory.threshold_filter(
                hls_ts, valid, class_label)

            return filtered

        # apply spectral filter to input labels: in parallel
        valid = Parallel(n_jobs=-1, verbose=51)(delayed(_filter)(labels, label) for label in class_labels)
        # for label in class_labels:
            #_filter(labels, label)
        

        # write spectrally filtered pixels to array of valid pixels
        for i, label in enumerate(class_labels):
            samples[np.where(valid[i])] = label.id

        return samples

    @staticmethod
    def percent_masked(hls_ts):
        """Compute proportion of invalid observations.

        Parameters
        ----------
        hls_ts : :py:class:`xarray.Dataset`
            The HLS time series.

        Returns
        -------
        :py:class:`xarray.DataArray`
            Proportion of invalid observations for each pixel.

        """
        return (hls_ts.qbin.sum(dim='time') / len(hls_ts.time)).compute()

    @staticmethod
    def read_labels(labels, label_name):
        """Translate input land cover product to classification legend.

        Parameters
        ----------
        labels : `str` or :py:class:`pathlib.Path`
            Path to the input land cover product.
        label_name : `str`
            Name of the input land cover product.

        Returns
        -------
        :py:class:`numpy.ndarray`
            The translated input land cover product.

        """
        return array_replace(img2np(labels),
                             SUPPORTED_LC_PRODUCTS[label_name].to_numpy())

    @staticmethod
    def apply_qa(hls_ts):
        """Apply the quality assessment layers.

        Parameters
        ----------
        hls_ts : :py:class:`xarray.Dataset`
            The HLS time series.

        Returns
        -------
        :py:class:`xarray.Dataset`
            The masked HLS time series.

        """
        return hls_ts.where(~hls_ts.qbin.astype(np.bool), other=np.nan)

    @staticmethod
    def hls_scale(hls_ts, tile_size=TILE_SIZE):
        """Scale the HLS time series to physical units.

        Parameters
        ----------
        hls_ts : :py:class:`xarray.Dataset`
            The HLS time series.
        tile_size : `tuple`, optional
            Chunk size to parallelize image operations using Dask. The default
            is ``ai4ebv.main.config.TILE_SIZE``.

        Returns
        -------
        hls_ts : :py:class:`xarray.Dataset`
            The scaled HLS time series.

        """
        # HLS multispectral time series
        hls_ts = hls_ts.chunk({'x': tile_size[0], 'y': tile_size[1]})

        # scale to physical units: dtype=Float32
        LOGGER.info('Scaling dataset to physical units ...')
        hls_ts = hls_ts.astype(np.float32, copy=False)
#        hls_ts = hls_ts.where(hls_ts != hls_ts.nodata, other=np.nan)
#        for name, v in hls_ts.data_vars.items():
            # skip the quality assessment layer: does not have a scale
#            if name not in ['qa', 'qbin']:
#                hls_ts[name] = v * hls_ts.scale

        return hls_ts

    @staticmethod
    def spectral_indices(hls_ts):
        """Compute spectral indices.

        Parameters
        ----------
        hls_ts : :py:class:`xarray.Dataset`
            The HLS time series.

        Returns
        -------
        ds : :py:class:`xarray.Dataset`
            The HLS time series with an additional set of spectral indices.

        """
        # check input type: either xarray.Dataset or numpy.ndarray
        LOGGER.info('Computing spectral indices: {}.'.format(
            ', '.join(INDEX_FEATURES)))

        # calculate spectral indices lazily
        with warnings.catch_warnings():
            # catch RunTimeWarning associated with zero divisions
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            ds = hls_ts.assign(
                {'ndvi': normalized_difference(hls_ts.nir, hls_ts.red),
                 'ndsi': normalized_difference(hls_ts.green, hls_ts.swir1),
                 'ndbi': normalized_difference(hls_ts.swir1, hls_ts.nir),
                 'ndwi': normalized_difference(hls_ts.green, hls_ts.nir),
                })

        # shape: (bands, time, y, x)
        return ds

    @staticmethod
    def is_homogeneous(arr, kernel_size=3):
        """Convolution for the erosion of boundary pixels.

        Parameters
        ----------
        arr : :py:class:`numpy.ndarray`
            The array to convolve.
        kernel_size : `int`, optional
            The convolutional kernel size. The default is `5`.

        Returns
        -------
        :py:class:`numpy.ndarray`
            The convolved array.

        """
        counts = convolve2d(arr, np.ones(2 * (kernel_size, )), mode='same',
                            boundary='fill', fillvalue=0)
        print(arr)
        return counts == (kernel_size ** 2)

    @staticmethod
    def spectral_distance(ds, dim='z'):
        """Compute the distance to the spectral class centroids.

        Parameters
        ----------
        ds : :py:class:`xarray.Dataset` or :py:class:`numpy.ndarray`
            The HLS time series.
        dim : `str`, optional
            The dimension to compute the distances along. The default is `'z'`.

        Returns
        -------
        distance : :py:class:`xarray.DataArray` or :py:class:`numpy.ndarray`
            Distance to the spectral class centroids.

        """
        # compute distance to spectral centroid
        with warnings.catch_warnings():
            # ignore RunTimeWarnings
            warnings.simplefilter('ignore', category=RuntimeWarning)

            # check input type: xr.Dataset
            if isinstance(ds, xr.Dataset):
                # compute median and mean absolute deviation (MAD) of spectral
                # bands at each time step: spectral centroid
                med_ds = ds.median(dim=dim)
                mad_ds = np.abs(ds - med_ds).median(dim='z')

                # compute distance to spectral centroid in each spectral band
                # for each time step: in units of MAD
                distance = np.abs((ds - med_ds) / mad_ds)

                # compute mean distance to spectral centroid in each spectral
                # band over time
                distance = distance.mean(dim='time', skipna=True)

                # compute mean distance over all spectral bands
                distance = distance.to_array().mean(axis=0)

            # check input type: np.array
            else:
                # compute median and mean absolute deviation (MAD) of spectral
                # bands at each time step: spectral centroid
                ds = np.asarray(ds)
                med_ds = np.nanmedian(ds, axis=-1, keepdims=True)
                mad_ds = np.nanmedian(np.abs(ds - med_ds), axis=-1,
                                      keepdims=True)

                # compute distance to spectral centroid in each spectral band
                # for each time step: in units of MAD
                distance = np.abs((ds - med_ds) / mad_ds)

                # compute mean distance to spectral centroid in each spectral
                # band over time
                distance = np.nanmean(distance, axis=(0, 1))

        return distance

    @staticmethod
    def distance_to_centroid(train_ds, mask, quantile=0.9, window_size=None):
        """Moving window algorithm for the spectral outlier detection.

        Parameters
        ----------
        train_ds : :py:class:`xarray.Dataset`
            The HLS time series.
        mask : :py:class:`numpy.ndarray`
            Mask of land cover class labels.
        quantile : `float`, optional
            Quantile above which the distance from the spectral centroid is
            considered as an outlier. The default is `0.9`.
        window_size : `int`, optional
            Size of the moving window. The default is `915`.

        Returns
        -------
        valid : py:class:`numpy.ndarray`
            Mask of filtered class labels.

        """
        # initialize boolean mask for valid pixels
        valid = np.zeros(mask.shape, dtype=np.int16)
        
        # check whether to calculate centroids on local windows
        if window_size is not None:
            # reshape inputs to local windows
            window_coordinates = tile_topleft_corner(mask.shape, window_size)
            mask = img2np(mask, tile_size=window_size, pad=False)
            valid = img2np(valid, tile_size=window_size, pad=False)
            # split time series to local windows
            ds = train_ds.to_array().chunk({'x': window_size,
                                            'y': window_size}).data

            # iterate over local windows
            for idx, pos in window_coordinates.items():

                # current position in array
                row, col = [int(p / window_size) for p in pos]

                # pixels within current local windows
                l_rows, l_cols = np.where(mask[idx, ...])
                if l_rows.size > 0:
                    # read pixels within current local window to memory
                    window = ds.blocks[..., row, col].compute()
                    window = window[..., l_rows, l_cols]

                    # compute spectral centroid and distances
                    distance = TrainingDataFactory.spectral_distance(window)

                    # apply threshold for valid pixels: distance < quantile
                    selected = distance < np.quantile(distance, quantile)

                    # write valid pixels to boolean mask
                    valid[idx, l_rows[selected], l_cols[selected]] = 1

            # reshape local windows to original shape
            valid = reconstruct_scene(valid)

        else:
            # rows and columns where mask is True
            rows, cols = np.where(mask)
            

            # create xarray dataarrays to select pixels of current class
            rows, cols = (xr.DataArray(rows, dims='z'),
                          xr.DataArray(cols, dims='z'))
            
        
            # select pixels of current class
            ds = train_ds.sel(x=cols, y=rows)

            # compute spectral centroid and distances
            distance = TrainingDataFactory.spectral_distance(ds)
            
            valid_pixel = distance < distance.quantile(quantile)
            
            if len(valid_pixel) == 0:
                return valid
            

            # apply threshold for valid pixels: distance < quantile
            distance = distance.where(valid_pixel,
                                      drop=True)
        

            # write valid pixels to boolean mask
            valid[distance.y, distance.x] = 1
        # return a boolean mask containing only valid pixels
        return valid

    @staticmethod
    def threshold_filter(hls_ts, mask, label):
        """Supervised threshold-based spectral filter for selected classes.

        Parameters
        ----------
        hls_ts : :py:class:`xarray.Dataset`
            The HLS time series.
        mask : :py:class:`numpy.ndarray`
            Mask of land cover class labels.
        label : :py:class:`enum.EnumMeta`
            Label of the land cover class.

        Returns
        -------
        py:class:`numpy.ndarray`
            Mask of filtered class labels.

        """
        # check if input class label is supported by the threshold filter
        if label not in SUPERVISED_CLASSES:
            return mask

        # initialize mask of valid pixels
        valid = np.zeros(mask.shape, dtype=np.int16)

        # rows and columns where mask is True
        rows, cols = np.where(mask)

        # create xarray dataarrays to select pixels of current class
        rows, cols = (xr.DataArray(rows, dims='z'),
                      xr.DataArray(cols, dims='z'))

        # select pixels of current class
        ds = hls_ts.sel(x=cols, y=rows)

        # compute spectral indices and drop spectral bands not required
        ds = TrainingDataFactory.spectral_indices(ds)
        ds = ds.drop_vars([b for b in USE_BANDS if b in ds.data_vars and b !=
                           'nir'])

        # threshold-based filter
        with warnings.catch_warnings():
            # catch RunTimeWarning associated with zero divisions
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            # surface water
            if label is Legend.Water_bodies:
                # NDWI > 0: NDWI > 0 for surface water
                condition = ds.ndwi.min(dim='time', skipna=True) > 0

            # snow and ice
            if label is Legend.Glaciers:
                # NDSI > 0.4 & NIR > 0.11: SNOWMAP algorithm after
                #                          Hall et al. (1995)
                condition = ((ds.ndsi.min(dim='time', skipna=True) > 0.4) &
                             (ds.nir.min(dim='time', skipna=True) > 0.11))

        # filter label by threshold
        filtered = ds.where(condition, drop=True)

        # boolean mask of valid pixels fulfilling threshold conditions
        valid[filtered.y, filtered.x] = 1

        return valid

    @staticmethod
    def features(hls_ts, percentiles, seasonal=False):
        """Compute spectral-temporal features.

        Parameters
        ----------
        hls_ts : :py:class:`xarray.Dataset`
            The HLS time series.
        percentiles : `list`[`float`]
            The percentiles of the time series.
        seasonal : `bool`, optional
            Whether to compute features on a seasonal (`True`) or annual
            (`False`) basis. The default is `False`.

        Returns
        -------
        ds : :py:class:`xarray.Dataset`
            The spectral-temporal features.

        """
        # feature engineering
        if seasonal:
            # percentiles of the spectral bands and indices for each season
            # DJF, MAM, JJA, SON
            ds = hls_ts.groupby('time.season').quantile(
                    percentiles, dim='time', skipna=False).rename(
                    {'quantile': 'features'})
        else:
             # percentiles of the spectral bands and indices over the entire
             # year
             ds = hls_ts.quantile(
                 percentiles, dim='time', skipna=False).rename(
                     {'quantile': 'features'})

        # shape: (bands, ..., y, x)
        return ds

    @staticmethod
    def add_coordinates(array, dims=('time', 'y', 'x')):
        """Add coordinates to an array to create a :py:class:`xarray.Dataset`.

        Parameters
        ----------
        array : :py:class:`numpy.ndarray`
            The array to annotate.
        dims : `tuple` [`str`], optional
            Dimension names. The default is `('time', 'y', 'x')`.

        Returns
        -------
        dims : `tuple` [`str`]
            Dimension names.
        array : :py:class:`numpy.ndarray`
            The input array.

        """
        return (dims, array)

    @staticmethod
    def repeat_along_axis(array, repeats, axis):
        """Repeat an array along an axis with Dask.

        Parameters
        ----------
        array : :py:class:`numpy.ndarray`
            The array to repeat.
        repeats : `int`
            Number of repeats.
        axis : `int`
            The axis along which to repeat.

        Returns
        -------
        :py:class:`dask.array.core.Array`
            The repeated array.

        """
        return da.repeat(da.asarray(array), repeats, axis=axis)

    @staticmethod
    def dem_features(dem, tile_size=TILE_SIZE, add_coord=None):
        """Compute DEM slope and aspect using GDAL Dem.

        Parameters
        ----------
        dem : `str` or :py:class:`pathlib.Path`
            Path to the digital elevation model.
        tile_size : `tuple`, optional
            Chunk size to parallelize DEM operations using Dask. The default
            is ``ai4ebv.main.config.TILE_SIZE``.
        add_coord : `dict`, optional
            Add a coordinate to the DEM, e.g. repeat along time. The default is
            `None`. If specified, a dictionary with ``key``: coordinate name,
            ``value``: coordinate values.

        Returns
        -------
        :py:class:`xarray.Dataset`
            The DEM elevation, slope and aspect.

        """
        def _scale(array):
            return minmax_scale(array.reshape(-1, 1)).reshape(array.shape)

        # read digital elevation model
        LogConfig.init_log('Reading digital elevation model: {}'.format(dem))

        # compute slope
        LOGGER.info('Computing terrain slope ...')
        slope = gdal.DEMProcessing(
            '', str(dem), 'slope', format='MEM', computeEdges=True, alg='Horn',
            slopeFormat='degrees').ReadAsArray().astype(np.float32)

        # compute aspect
        LOGGER.info('Computing terrain aspect ...')
        aspect = gdal.DEMProcessing(
            '', str(dem), 'aspect', format='MEM', computeEdges=True,
            alg='Horn').ReadAsArray().astype(np.float32)

        # scale digital elevation model to range [0-1]
        dem = _scale(img2np(dem).astype(np.float32))
        slope = _scale(slope)
        aspect= _scale(aspect)

        # digital elevation model features: elevation, slope and aspect
        dem_vars = {
            'dem': TrainingDataFactory.add_coordinates(dem, ('y', 'x')),
            'slope': TrainingDataFactory.add_coordinates(slope, ('y', 'x')),
            'aspect': TrainingDataFactory.add_coordinates(aspect, ('y', 'x'))}

        # create xarray.Dataset for digital elevation model
        dem_features = xr.Dataset(data_vars=dem_vars,
                                  coords={'y': np.arange(0, dem.shape[0]),
                                          'x': np.arange(0, dem.shape[1])})

        # check whether to add additional coordinates
        if add_coord is not None:
            for var in dem_features.data_vars:
                # expand variable alogn new dimension
                for k, v in add_coord.items():
                    expanded = TrainingDataFactory.repeat_along_axis(
                        np.expand_dims(dem_features[var], axis=0),
                        len(v), axis=0)

                    # overwrite variable
                    dem_features[var] = TrainingDataFactory.add_coordinates(
                        expanded, (k, 'y', 'x'))

            # assign new coordinate
            dem_features = dem_features.assign_coords(add_coord)

        return dem_features

    @staticmethod
    def spatially_stratified_sampling(samples, npixel, label, buffer=None,
                                      apriori=False):
        """Random spatially stratified training data sampling.

        Parameters
        ----------
        samples : :py:class:`numpy.ndarray`
            Input class labels.
        npixel : `int`
            Number of samples to select for each class.
        label : `int`
            Value of the class label in ``samples``.
        buffer : `int`, optional
            Distance between training samples. The default is `None`.
        apriori : `bool`, optional
            Whether to select an equal amount of training pixels according to
            ``npixel`` (`False`) or to select based on the distribution of the
            classes in ``samples`` (`True`). The default is `False`.

        Returns
        -------
        selected : :py:class:`numpy.ndarray`
            The selected training pixels.

        """
        # initiate training data sampling for current class
        selected = np.zeros(samples.shape)

        # get the indices of the current class in the tile
        mask = samples == label
        rows, cols = np.where(mask)

        # number of available pixels for current class
        available = np.count_nonzero(mask)

        # number of pixels to sample
        if apriori:
            # number of valid samples: exclude pixels not selected by spectral
            # thresholding
            valid_samples = np.count_nonzero(samples != Legend.NoData.id)

            # number of samples proportional to area covered by the class
            to_sample = int((available / valid_samples) * npixel)
        else:
            # number of samples equal to constant: class-wise stratification
            to_sample = min(available, npixel)

        # buffer: number of pixels defining radius around a training pixel,
        #         within which no other training pixel should be sampled
        if buffer is None or buffer == 0:
            # randomly choose npixel from the available pixels
            indices = np.random.choice(
                available, size=to_sample, replace=False)

            # row and column of the selected pixel
            row, col = rows[indices], cols[indices]
            selected[row, col] = 1
        else:
            # maximum number of iterations: inversely proportional to squared
            # buffer size, required to limit computational cost
            max_iter = int(npixel / (buffer ** 2))

            # sample with constraint of buffer size
            counter = 0
            while np.count_nonzero(selected) < to_sample:
                # check remaining pixels
                if not rows.size > 0:
                    break

                # check number of iterations
                if counter >= max_iter:
                    # reset available pixels: re-initialize sampling for
                    # smaller buffer size
                    rows, cols = np.where(mask)
                    available = np.count_nonzero(mask)

                    # reduce buffer size by one pixel and reset counter
                    buffer, counter = buffer - 1, 0

                # randomly choose a pixel from the available pixels
                idx = np.random.choice(available, size=1, replace=False).item()

                # row and column of the selected pixel
                row, col = rows[idx], cols[idx]

                # check whether sampled training pixel is located at least
                # buffer pixels from other training pixels
                neighborhood = selected[(row - buffer):(row + buffer + 1),
                                        (col - buffer):(col + buffer + 1)]
                if neighborhood.sum() < 1:
                    # sampled pixel is located at least buffer pixels from
                    # any other sampled pixel
                    selected[row, col] = 1
                else:
                    # advance counter, if sampled pixel is located in a
                    # neighborhood
                    counter += 1

                # remove sampled pixel and update available pixels
                rows = np.delete(rows, idx, axis=0)
                cols = np.delete(cols, idx, axis=0)
                available -= 1

        return selected

    @staticmethod
    def generate_training_data(hls_ts, samples, labels, N, layer_only=False,
                               no_data=Legend.NoData.id, **kwargs):
        """Retrieve the spectra of the training data from the HLS time series.

        Parameters
        ----------
        hls_ts : :py:class:`xarray.Dataset`
            The HLS time series.
        samples : :py:class:`numpy.ndarray`
            Input class labels.
        labels : :py:class:`enum.EnumMeta`
            The land cover classification legend.
        N : `int`
            Number of samples to select for each class.
        layer_only : `bool`, optional
            Whether to return both the position and spectra of the training
            pixels (`False`) or only the position (`True`). The default is
            `False`.
        no_data : `int`, optional
            Value indicating missing class labels. The default is
            :py:class:`ai4ebv.core.legend.Legend.NoData.id`.
        **kwargs : `dict`
            Additional keyword arguments passed to :py:meth:`ai4ebv.core.
            sample.TrainingDataFactory.spatially_stratified_sampling`.

        Returns
        -------
        ds : :py:class:`xarray.Dataset`
            The training dataset spectra.
        layer : :py:class:`numpy.ndarray`
            The spatial position of the training pixels.

        """
        # initialize logging of training data selection
        LogConfig.init_log('Spatially stratified sampling.')

        # replace missing values in time series by median
        hls_ts = hls_ts.where(~np.isnan(hls_ts),
                              other=hls_ts.median(dim='time', skipna=True))

        # create the training data image: layer of selected training pixels
        layer = (np.ones((len(hls_ts.x), len(hls_ts.y)), dtype=np.int16) *
                 no_data)

        # run spatially stratified training data sampling in parallel
        selected = Parallel(n_jobs=-1, verbose=51)(
            delayed(TrainingDataFactory.spatially_stratified_sampling)(
                samples, N, label.id, **kwargs) for label in labels)

        # fill the training data layer
        for label, sel in zip(labels, selected):
            layer[np.where(sel)] = label.id

        # check whether to return only position of training data or position
        # and spectra
        training_data = []
        if not layer_only:
            # helper function for selecting training data in parallel
            def _retrieve_spectra(hls_ts, label, mask):
                # indices of the mask
                rows, cols = np.where(mask)

                # log how many pixels have been sampled for the class
                LOGGER.info('Sampled {} pixels for class: {}'.format(
                        np.count_nonzero(mask), label.name))

                # check whether the current class is present in the image
                if not rows.size:
                    return

                # create xarray dataarrays to select pixels of current class
                rows, cols = (xr.DataArray(rows, dims='samples'),
                              xr.DataArray(cols, dims='samples'))

                # catch RuntimeWarnings of dask: clutters logging
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)

                    # time series of the selected pixels
                    ds = hls_ts.sel(x=cols, y=rows)

                    # add label to fields
                    label_field = da.ones((len(ds.time), len(ds.samples)),
                                          dtype=np.int16) * label.id
                    ds = ds.assign({
                        'label': TrainingDataFactory.add_coordinates(
                            label_field, dims=('time', 'samples'))})

                return ds

            # retrieve spectra of training data
            LogConfig.init_log('Retrieving training data spectra.')
            training_data = []
            for label, mask in zip(labels, selected):
                training_data.append(_retrieve_spectra(hls_ts, label, mask))

            # remove classes for which no valid pixels are found
            training_data = list(filter(None, training_data))

        # merge training data of different classes
        ds = xr.concat(training_data, dim='samples') if training_data else None

        return ds, layer

    @staticmethod
    def load_training_data(training_data, features=False, percentiles=None,
                           use_indices=False, seasonal=False, no_data=-1):
        """Load the training dataset to memory.

        Parameters
        ----------
        training_data : :py:class:`pathlib.Path` or :py:class:`xarray.Dataset`
            Path to or instance of the training dataset.
        features : `bool`, optional
            Whether to compute spectral-temporal features (`True`) or return
            the time series of the spectra (`False`). The default is `False`
        percentiles : `list` [`float`], optional
            The percentiles of the time series. The default is `None`.
        use_indices : `bool`, optional
            Whether to compute spectral indices. The default is `False`.
        seasonal : `bool`, optional
            Whether to compute features on a seasonal (`True`) or annual
            (`False`) basis. The default is `False`.
        no_data : `float`, optional
            Value to assign to missing observations. The default is `-1`.

        Returns
        -------
        inputs : :py:class:`numpy.ndarray`
            The training data spectra.
        labels : TYPE
            The training data class labels.

        """
        # read training data
        LogConfig.init_log('Loading training data to memory.')
        if isinstance(training_data, xr.Dataset):
            ds = training_data
        else:
            ds = xr.open_dataset(training_data)

        # get labels of the training data: constant over time
        labels = ds.label.isel(time=0).values
        ds = ds.drop_vars('label')

        # whether to use spectral indices for classification
        if use_indices:
            ds = TrainingDataFactory.spectral_indices(ds)

        # whether to use the raw time series or derive classification features
        if not features:
            # reshape: (nsamples, bands, length)
            inputs = ds.to_array().values.swapaxes(0, -1).swapaxes(1, -1)
        else:
            # generate classification features
            if percentiles is None:
                percentiles = TrainingDataFactory.percentiles

            # shape: (bands, ..., nsamples)
            inputs = TrainingDataFactory.features(ds, percentiles,
                                                  seasonal=seasonal).to_array()

            # reshape: (nsamples, seasons, features, bands)
            inputs = inputs.values.swapaxes(0, -1)

            # reshape: (nsamples, nfeatures)
            inputs = inputs.reshape(inputs.shape[0], -1)

        # mask pixels for which not a single observation is valid
        inputs = np.nan_to_num(inputs, nan=no_data, posinf=no_data,
                               neginf=no_data)

        # check if all pixels are valid and finite
        assert ((~np.isnan(inputs).any()) & (np.isfinite(inputs).all()))

        return inputs, labels
