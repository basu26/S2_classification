"""Dataset classes compliant to the Pytorch standard."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import urllib
import pathlib
import logging
import calendar
import datetime
import subprocess
from joblib import Parallel, delayed

# externals
import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
from osgeo import gdal
from torch.utils.data import Dataset

# locals
from pysegcnn.core.utils import (search_files, img2np, array_replace, doy2date,
                                 read_hdf4, HIERARCHICAL_DATA_FORMAT)
from pysegcnn.core.trainer import LogConfig
from ai4ebv.core.constants import (S2BANDS, L8BANDS, L8SUBDATASETS,
                                   S2SUBDATASETS, USE_BANDS, QA_TO_MASK)
from ai4ebv.core.qa import qa2binary
from ai4ebv.core.constants import OLABELS, HLS_TILE, HLS_DATE
from ai4ebv.main.config import HERE
from ai4ebv.main.io import HLS_PATH

# module level logger
LOGGER = logging.getLogger(__name__)


class EoDataset(Dataset):
    """Base dataset class."""

    def __init__(self, root_dir):
        """Initialize.

        Parameters
        ----------
        root_dir : `str` or :py:class:`pathlib.Path`
            Path to the dataset.

        """
        super().__init__()

        # instance attributes
        self.root_dir = root_dir

    @staticmethod
    def to_tensor(x, dtype):
        """Convert ``x`` to :py:class:`torch.Tensor`.

        Parameters
        ----------
        x : array_like
            The input data.
        dtype : :py:class:`torch.dtype`
            The data type used to convert ``x``.

            The modified class labels.
        Returns
        -------
        x : `torch.Tensor`
            The input data tensor.

        """
        return torch.tensor(np.asarray(x).copy(), dtype=dtype)

    @staticmethod
    def transform_gt(labels, original_labels=None, invert=False):
        """Map the original class labels to the model class labels.

        Modify the values of the class labels in the ground truth mask to match
        the PyTorch standard. The class labels should be represented by an
        increasing sequence of integers starting from 0.

        Parameters
        ----------
        labels : :py:class:`numpy.ndarray`
            The class labels.
        original_labels: `list` [`int`], optional
            List of unique class labels to transform. If `None`, ``labels`` is
            assumed to contain all unique class labels.
        invert : `bool`, optional
            Whether to convert from original to PyTorch (``invert=False``) or
            from PyTorch to original class labels (``invert=True``). The
            default is `False`.

        Returns
        -------
        gt : :py:class:`numpy.ndarray`
            The model class labels.

        """
        # convert to numpy array
        labels = np.expand_dims(np.asarray(labels), axis=0)

        # original class labels
        if original_labels is None:
            original_labels = np.unique(labels)
        else:
            original_labels = np.asarray(original_labels)

        # new class labels
        new_labels = np.arange(0, len(original_labels))

        # replace the original labels by an increasing sequence of class labels
        if not invert:
            # define label map from original labels to PyTorch
            lookup = np.stack([original_labels, new_labels], axis=1)
        else:
            # define label map from PyTorch to original labels
            lookup = np.stack([new_labels, original_labels], axis=1)

        # convert labels
        labels = array_replace(labels, lookup)

        return labels.squeeze()


class HLSMetaDataset(EoDataset):
    """Base class for the `Harmonized Landsat Sentinel-2`_ dataset.

    .. _Harmonized Landsat Sentinel-2:
        https://hls.gsfc.nasa.gov/

    """
    # class attributes
    resolution = 30  # spatial resolution
    nodata = -1000  # NoData value
    scale = 0.0001  # scaling factor
    bands = USE_BANDS  # multispectral bands of interest
    qa_params = QA_TO_MASK  # quality assessment parameters to mask
    azure = False  # whether the dataset is hosted in the Azure Blob Storage
    ftp = 'https://hls.gsfc.nasa.gov/data/'  # Nasa's ftp url hosting HLS data

    # compression for netcdf time series
    comp = dict(dtype='int16', complevel=5)

    def __init__(self, root_dir):
        """Initialize.

        Parameters
        ----------
        root_dir : `str` or :py:class:`pathlib.Path`
            Path to the dataset.

        """
        # initialize EoDataset class
        super().__init__(root_dir)

    @property
    def s2bands(self):
        """Sentinel-2 band numbers.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Sentinel-2 band numbers.

        """
        return np.array([S2BANDS.index(b) for b in self.bands] if self.bands
                        else np.arange(0, len(S2BANDS)))

    @property
    def l8bands(self):
        """Landsat-8 band numbers.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Landsat-8 band numbers.

        """
        return np.array([L8BANDS.index(b) for b in self.bands] if self.bands
                        else np.arange(0, len(L8BANDS)))

    def is_sentinel(self, scene_name):
        """Check if a HLS scene is acquired by Sentinel-2 or Landsat.

        Parameters
        ----------
        scene_name : `str` or :py:class:`pathlib.Path`
            Name of the HLS scene.

        Returns
        -------
        `bool`
            `True` if scene is acquired by Sentinel-2, `False` if by Landsat.

        """
        return str(scene_name).startswith('HLS.S')

    def use_bands(self, scene_name):
        """Spectral bands of a scene.

        Parameters
        ----------
        scene_name : `str` or :py:class:`pathlib.Path`
            Name of the HLS scene.

        Returns
        -------
        :py:class:`numpy.ndarray`
            Spectral bands of the scene ``scene_name``.

        """
        return self.s2bands if self.is_sentinel(scene_name) else self.l8bands

    @staticmethod
    def parse_tile(hls_scene):
        """Parse the tile name of a HLS scene.

        Parameters
        ----------
        hls_scene : `str` or :py:class:`pathlib.Path`
            Name of the HLS scene.

        Returns
        -------
        `str`
            Name of the HLS tile.

        """
        return HLS_TILE.search(str(hls_scene))[0].lstrip('T')

    @staticmethod
    def parse_date(hls_scene):
        """Parse the date of a HLS scene.

        Parameters
        ----------
        hls_scene : `str` or :py:class:`pathlib.Path`
            Name of the HLS scene.

        Returns
        -------
        :py:class:`datetime.datetime`
            Date of the HLS scene.

        """
        ydoy = HLS_DATE.search(str(hls_scene))[0]
        return doy2date(int(ydoy[:4]), int(ydoy[4:]))

    @staticmethod
    def parse_doy(hls_scene):
        """Parse the day of the year of a HLS scene.

        Parameters
        ----------
        hls_scene : `str` or :py:class:`pathlib.Path`
            Name of the HLS scene.

        Returns
        -------
        `str`
            Day of the year of the HLS scene.

        """
        ydoy = HLS_DATE.search(str(hls_scene))[0]
        return ydoy


class HLSDataset(HLSMetaDataset):
    """Base class to process HLS data."""

    def __init__(self, root_dir, tile, year, months=[], hdf=True):
        """Initialize a HLS dataset.

        Parameters
        ----------
        root_dir : `str` or :py:class:`pathlib.Path`
            Path to the HLS scenes.
        tile : `str`
            Name of the HLS tile.
        year : `int`
            Reference year.
        months : array-like, optional
            Months of interests. The default is `[]`, which means
            using HLS scenes for the entire reference year.
        hdf : `bool`, optional
            Whether the HLS scenes are stored in the Hierarchical Data Format
            (hdf). The default is `True`. Set ``hdf=False`` if working with
            GeoTiff files.

        """
        super().__init__(root_dir)

        # instance attributes
        self.tile = tile.lstrip('T')  # tile of interest
        self.year = year  # year(s) of interest
        self.hdf = hdf  # hls data format: hdf or tif

        # month(s) of interest
        self.months = np.asarray(months) if months else np.arange(1, 13)

        # available hls-scenes matching the specified tile, year and months
        self.scenes = self.compose_scenes()
        if self.scenes:
            # sort scenes in chronological order
            LOGGER.info('Sorting scenes in chronological order ...')
            self.scenes = sorted(self.scenes, key=lambda x: self.parse_date(x))
            LOGGER.info('Found HLS-scenes:')
            LOGGER.info(('\n ' + (len(__name__) + 1) * ' ').join(
                ['{}'.format(str(s)) for s in self.scenes]))

            # check if months are specified
            if len(self.months) < 12:
                LOGGER.info('Using scenes from months: {}.'.format(', '.join(
                    [calendar.month_name[m] for m in self.months])))
                # drop scenes which are not in specified months
                self.scenes = [s for s in self.scenes if
                               self.parse_date(s).month in self.months]

    def __len__(self):
        """Number of scenes in the HLS dataset."""
        return len(self.scenes)

    def compose_scenes(self):
        """Method to build the dataset.

        Raises
        ------
        NotImplementedError
            Inherit the HLSDataset class and implement this method.

        """
        raise NotImplementedError('Inherit {} and implement this method.'.
                                  format(self.__class__.__name__))

    @staticmethod
    def time_series(tile, year, months):
        """Generates a default filename for a HLS time series dataset.

        Parameters
        ----------
        tile : `str`
            Name of the HLS tile.
        year : `int`
            Reference year.
        months : array-like
            Months of interests.

        Returns
        -------
        `str`
            Filename: S2_<tile>_<year>_<start-month>_<end-month>.nc

        """
        # time series name: HLS_<tile>_<year>_<start-month>_<end-month>.nc
        months = np.asarray(months) if months else np.arange(1, 13)
        return '_'.join(['S2', str(tile), str(year),
                         calendar.month_name[months[0]],
                         calendar.month_name[months[-1]]]) + '.nc'

    @staticmethod
    def state_file(model, labels, npixel, features=True, mode='single',
                   months=None, dem=False, qa=False, suffix=None,
                   use_indices=False, seasonal=False, year=None):
        """Generates a default filename for a classifier.

        Parameters
        ----------
        model : `str`
            Name of the classifier.
        labels : `str`
            Name of the input land cover product.
        npixel : `int`
            Number of training data pixels for each class.
        features : `bool`, optional
            Whether the classifier uses spectral-temporal features or the raw
            time series. The default is `True`.
        mode : `str`, optional
            Classification mode, either `'single'` or `'batch'`. The default is
            `'single'`.
        months : array-like, optional
            Months of interest. The default is None.
        dem : `bool`, optional
            Whether the classifier uses digital elevation model features. The
            default is `False`.
        qa : `bool`, optional
            Whether to apply the HLS quality assessment layers. The default is
            `False`.
        suffix : `str`, optional
            Arbitrary suffix to append to filename. The default is `None`.
        use_indices : `bool`, optional
            Whether the classifier uses spectral indices. The default is
            `False`.
        seasonal : `bool`, optional
            Whether the classifier uses seasonal (`True`) or annual (`False`)
            spectral-temporal features. The default is `False`.
        year : `int`, optional
            Reference year. The default is `None`.

        Returns
        -------
        `str`
            Classifier filename.

        """
        # check whether time series or classification features are used
        state = '_'.join([model, mode, 'FT' if features else 'TS'])

        # reference year
        if year is not None:
            state = '_'.join([state, str(year)])

        # months of considered images
        if months is not None:
            state = '_'.join([state, 'M{}'.format(''.join([str(m) for m in
                                                           months]))])

        # check which labels are used
        state = '_'.join([state, labels])

        # check whether specral indices are used
        state = '_'.join([state, 'IND']) if use_indices else state

        # check whether seasonal or annual features are used
        state = '_'.join([state, 'SNL' if seasonal else 'ANN'])

        # check whether digital elevation model is used
        state = '_'.join([state, 'DEM']) if dem else state

        # check whether the quality assessment layer is applied
        state = '_'.join([state, 'QA']) if qa else state

        # number of pixels
        state = '_'.join([state, 'N{}'.format(npixel)])

        # check whether a suffix is specified
        state = '_'.join([state, suffix]) if suffix is not None else state

        return state + '.pt'

    def read_scene(self, scene, spatial_coverage, cloud_coverage):
        """Read a HLS scene to memory.

        Parameters
        ----------
        scene : `str` or :py:class:`pathlib.Path`
            Path to the HLS scene.
        spatial_coverage : `int` or `float`
            Minimum amount of spatial coverage for the scene to be considered
            as valid.
        cloud_coverage : `int` or `float`
            Maximum amount of cloud coverage for the scene to be considered as
            valid.

        Returns
        -------
        date : :py:class:`datetime.datetime`
            Date of the HLS scene.
        ds : :py:class:`numpy.ndarray`
            The data of the HLS scene.

        """
        # check whether reading from the Azure Blob Storage
        meta = scene
        if self.azure:
            meta = '/vsicurl/{}'.format(meta.replace('.tif', '_01.tif'))

        # gdalinfo: image metadata dictionary
        try:
            # try to read metadata from scene
            info = gdal.Info(str(meta), format='json')
        except:
            # in case of any error, skip scene
            return

        # get image metadata dictionary
        try:
            metadata = info['metadata']['']
        except KeyError:
            LOGGER.info('Cannot retrieve metadata for scene: {}'.format(scene))
            return

        # check spatial coverage
        sc = metadata.get('spatial_coverage')
        if sc is not None:
            # drop scene if spatial coverage less than defined threshold
            if int(sc) < spatial_coverage:
                LOGGER.info('Dropping scene: {}, Coverage < {:d}%'
                            .format(scene, spatial_coverage))
                return

        # check cloud coverage
        cc = metadata.get('cloud_coverage')
        if cc is not None:
            if int(cc) > cloud_coverage:
                LOGGER.info('Dropping scene: {}, Cloud cover > {:d}%'
                            .format(scene, cloud_coverage))
                return

        # scene passed both quality checks: used for further processing
        LOGGER.info('Processing scene: {}, Coverage: {}%, Cloud cover: {}%'
                    .format(scene, sc, cc))

        # read current scene to array
        ds = self[self.scenes.index(scene)]
        if ds is None:
            return

        # compute the binary quality assessment layer
        qbin = qa2binary(ds[self.bands.index('qa'), ...],
                         parameters=self.qa_params)
        ds = np.concatenate((ds, np.expand_dims(qbin, 0)), axis=0)
        del qbin

        return self.parse_date(scene), ds

    def to_xarray(self, target=None, spatial_coverage=10, cloud_coverage=50,
                  save=True, overwrite=False):
        """Read a time series of HLS scenes to an :py:class:`xarray.Dataset`.

        Parameters
        ----------
        target : `str` or :py:class:`pathlib.Path`, optional
            Path to save HLS time series. The default is `None`. If specified,
            the :py:class:`xarray.Dataset` is saved to ``target`` as NetCDF.
        spatial_coverage : `int` or `float`, optional
            Minimum amount of spatial coverage for the scene to be considered
            as valid. The default is `10`.
        cloud_coverage : `int` or `float`, optional
            Maximum amount of cloud coverage for the scene to be considered as
            valid. The default is `50`.
        save : `bool`, optional
            Whether to save the HLS time series to ``target``. The default is
            `True`.
        overwrite : `bool`, optional
            Whether to overwrite ``target``. The default is `False`.

        Returns
        -------
        ds : :py:class:`xarray.Dataset`
            The time series of the HLS scenes.

        """
        # iterate over the scenes of the dataset and check metadata
        # spatial coverage: percentage of HLS tile with data
        # cloud coverage: percentage of cloud cover per tile
        ds = Parallel(n_jobs=-1, verbose=51)(
            delayed(self.read_scene)(scene, spatial_coverage, cloud_coverage)
            for scene in self.scenes)

        # convert time series to numpy array
        ts = np.asarray([scene[1] for scene in ds if scene is not None])
        dates = np.asarray([scene[0] for scene in ds if scene is not None])

        # check if time series has at least one image
        if all([img is None for img in ts]):
            LOGGER.info('No image with SC>{}% and CC<{}% found for tile {} and'
                        'year {}.'.format(spatial_coverage, cloud_coverage,
                                          self.tile, self.year))
            return

        # create a dictionary of the different spectral bands
        ts_ds = {k: (('time', 'y', 'x'), ts[:, i, ...]) for i, k in
                 enumerate(self.bands + ['qbin'])}

        # create xarray dataset: dtype=Int16
        ds = xr.Dataset(data_vars=ts_ds,
                        coords={'time': dates,
                                'y': np.arange(0, ts.shape[2]),
                                'x': np.arange(0, ts.shape[3])},
                        attrs={'scale': self.scale,
                               'nodata': self.nodata,
                               'resolution': self.resolution
                               })

        # clear memory
        del ts, ts_ds

        # save dataset to disk
        if save and target is not None:
            target = pathlib.Path(target)
            LOGGER.info('Saving HLS time series: {}'.format(target))
            encoding = {var: self.comp for var in ds.data_vars}

            # check if output directory exists
            if not target.parent.exists():
                LOGGER.info('mkdir {}'.format(target.parent))
                target.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(target, encoding=encoding)

        return ds

    @staticmethod
    def download_hls(target, tile, year):
        """Download HLS data from NASA's ftp https://hls.gsfc.nasa.gov/data/.

        Parameters
        ----------
        target : `str` or :py:class:`pathlib.Path`
            Path to save the HLS data.
        tile : `str`
            Name of the HLS tile.
        year : `int`
            Reference year.

        """
        # search download bash script
        script = str(search_files(HERE.parent.parent, 'download.hls.sh').pop())

        # create target directory, if it does not exist
        target = pathlib.Path(target)
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)

        # bash download.hls.sh -t <tile> -y <year> TARGET
        command = [script, '-t', tile, '-y', str(year), str(target)]
        LogConfig.init_log('Downloading files from {} to {}.'
                           .format(HLSDataset.ftp, target))
        subprocess.run(command, stderr=subprocess.STDOUT)

    @staticmethod
    def check_scenes(hls, root=None, **kwargs):
        """Check if scenes are available for a HLS dataset.

        Parameters
        ----------
        hls : :py:class:`ai4ebv.core.dataset.HLSDataset`
            The HLS dataset.
        root : `str` or :py:class:`pathlib.Path`, optional
            Path to save the HLS data. The default is `None`. If specified and
            scenes are not available, the scenes are downloaded to ``root``.
        **kwargs : `dict`
            Additional keyword arguments passed to
            :py:class:`ai4ebv.core.dataset.HLSFileSystemDataset`.

        Returns
        -------
        hls : :py:class:`ai4ebv.core.dataset.HLSDataset`
            The HLS dataset.

        """
        # check whether the HLS data is available
        if not hls.scenes:
            LOGGER.info('Found no files for tile {} and year {} in {}.'
                        .format(hls.tile, hls.year, hls.root_dir))

            # download data to specified root directory
            if root is not None:
                # check whether the data already exist in the root directory
                hls = HLSFileSystemDataset(root, hls.tile, hls.year, **kwargs)
                if not hls.scenes:
                    # download data to root directory
                    HLSDataset.download_hls(root, hls.tile, hls.year)
                    hls = HLSFileSystemDataset(root, hls.tile, hls.year,
                                               **kwargs)
        return hls

    @staticmethod
    def initialize(root, tile, year, months, azure=False):
        """Initialize a HLS dataset.

        Parameters
        ----------
        root : `str` or :py:class:`pathlib.Path`
            Path to the HLS scenes.
        tile : `str`
            Name of the HLS tile.
        year : `int`
            Reference year.
        months : array-like, optional
            Months of interests.
        azure : `bool`, optional
            Whether to read the HLS data from the `Microsoft Azure container`_
            (`True`) or from a local filesystem (`False`). The default is
            `False`.

        Returns
        -------
        hls : :py:class:`ai4ebv.core.dataset.HLSDataset`
            The HLS dataset.

        .. _Microsoft Azure container:
            https://hlssa.blob.core.windows.net/hls

        """
        # check if in Azure cloud or on local cluster
        if azure:
            # read HLS-dataset from Azure Blob Storage
            hls = HLSAzureBlobStorage(tile, year, months=months, hdf=False)
        else:
            # read HLS-dataset from local file system
            hls = HLSFileSystemDataset(root, tile, year, months=months,
                                       hdf=True)

        # check whether scenes are found
        hls = HLSDataset.check_scenes(hls, root=HLS_PATH, months=months,
                                      hdf=True)

        return hls


class HLSFileSystemDataset(HLSDataset):
    """Base class for the HLS dataset stored on a filesystem."""

    # file naming convention in NASA's FTP
    pattern = 'HLS.(S|L)30.T{}.{}(.*)v1.4.{}$'

    def __init__(self, root_dir, tile, year, months=[], hdf=True):
        super().__init__(root_dir, tile, year, months, hdf)

    def compose_scenes(self):
        """Find the HLS scenes on the filesytem matching the configuration.

        Returns
        -------
        scenes : `list` [:py:class:`pathlib.Path`]
            List of the HLS scenes.

        """
        # construct file pattern to match
        pattern = self.pattern.format(self.tile, self.year, 'hdf' if self.hdf
                                      else 'tif')

        # search for hls-scenes matching the defined pattern
        scenes = search_files(self.root_dir, pattern)

        return scenes

    def __getitem__(self, idx):
        """Read a scene from the HLS dataset.

        Parameters
        ----------
        idx : `int`
            Index of the scene.

        Returns
        -------
        ds : :py:class:`numpy.ndarray`
            The data of the HLS scene.

        """
        # check file type of input scene: hdf4 or tif file
        file = self.scenes[idx]
        if self.hdf and file.suffix in HIERARCHICAL_DATA_FORMAT:
            # read hdf4 file: convert to same array shape
            ds = read_hdf4(file).to_array().values
            ds = img2np(ds)
        else:
            # read any image format supported by gdal
            ds = img2np(file)

        # subset to specified bands
        if ds.shape[0] > 1:
            ds = ds[self.use_bands(file.name), ...]

        return ds


class HLSAzureBlobStorage(HLSDataset):
    """Base class for the HLS dataset stored in a cloud container."""

    # set azure class attribute to True
    azure = True

    # HLS.<product>.T<tileid>.<doy>.v1.4_<subdataset>.tif
    pattern = 'HLS.{}.T{}.{}.v1.4_01.tif'  # file naming convention in Azure

    def __init__(self, tile, year, months=[], hdf=False,
                 container_url='https://hlssa.blob.core.windows.net/hls'):
        super().__init__(container_url, tile, year, months, hdf)

    def compose_scenes(self):
        """Find the HLS scenes in the cloud container.

        Returns
        -------
        scenes : `list` [`str`]
            List of the HLS scenes in the container.

        """
        # range of day(s) of year to search for
        s_day = datetime.date(self.year, self.months[0], 1).timetuple().tm_yday
        e_day = datetime.date(self.year, self.months[-1], calendar.monthlen(
            self.year, self.months[-1])).timetuple().tm_yday

        # iterate over the specified range of days and check which days exist
        scenes = []
        for doy in range(s_day, e_day):
            # Landsat-8 or Sentinel-2 product
            for product in ['L30', 'S30']:
                # scene to search for
                scene = '/'.join([self.root_dir, product, self.pattern.format(
                    product, self.tile.lstrip('T'),
                    ''.join([str(self.year), '{:03d}'.format(doy)]))])

                # check if the scene exists: get HTTP response code
                LOGGER.info('Searching: {}'
                            .format(scene.replace('_01.tif', '.tif')))
                try:
                    if urllib.request.urlopen(scene).getcode() == 200:
                        scenes.append(scene.replace('_01.tif', '.tif'))
                except:
                    continue

        return scenes

    def __getitem__(self, idx):
        """Read a scene from the HLS dataset.

        Parameters
        ----------
        idx : `int`
            Index of the scene.

        Returns
        -------
        ds : :py:class:`numpy.ndarray`
            The data of the HLS scene.

        """
        # url and path to current scene
        url = urllib.parse.urlparse(self.scenes[idx])
        scene = pathlib.Path(url.path)

        # iterate over the bands of interest
        band_data = []
        for band in self.bands:
            # get subdatset number: check whether the scene is S2 or L8
            band_number = (S2SUBDATASETS[band] if self.is_sentinel(scene.name)
                           else L8SUBDATASETS[band])

            # url to the current band
            band_url = url.geturl().replace('.tif', '_{}.tif'.format(
                band_number))

            # check if file exists
            if gdal.Open('/vsicurl/{}'.format(str(band_url))) is None:
                LOGGER.info('{} does not exist.'.format(band_url))
                return

            # get data of the current band
            band_data.append(img2np(band_url))

        # convert bands to numpy array
        return np.stack(band_data, axis=0)


class TabularDataset(EoDataset):
    """Base dataset for training neural network with PyTorch."""

    def __init__(self, inputs, labels):
        """Initialize dataset.

        Parameters
        ----------
        inputs : array-like, shape=(nsamples, ...)
            Input data.
        labels : array-like, shape=(nsamples,)
            Target labels.

        """
        # the tabular data: shape (nsamples, ...)
        self.inputs = [self.to_tensor(i, dtype=torch.float32) for i in inputs]

        # the class labels: shape (nsamples,)
        self.labels = [self.to_tensor(
            self.transform_gt(i, original_labels=OLABELS), dtype=torch.uint8)
            for i in labels]

    def __len__(self):
        """Number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Read a sample from the dataset.

        Parameters
        ----------
        idx : `int`
            The index of the sample.

        Returns
        -------
        input : :py:class:`torch.Tensor`
            Sample input data.
        label : :py:class:`torch.Tensor`
            Sample target label.

        """
        return self.inputs[idx], self.labels[idx]


class PadCollate(object):
    """Padding class for time series of different lengths.

    To be used with :py:class:`torch.utils.data.DataLoader` keyword argument
    ``collate_fn=PadCollate(dim)``.

    Modified from this PyTorch `thread`_.

    .. _thread:
        https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8

    """

    def __init__(self, dim=0):
        """Initialize padding class.

        Parameters
        ----------
        dim : `int`
            The dimension to be padded. The dimension of time for time series
            data.

        """
        # instance attribute
        self.dim = dim

    def pad_collate(self, batch, **kwargs):
        """Function collating training samples to mini-batches.

        Parameters
        ----------
        batch : `list` [`tuple`]
            List of tuples (input data, labels). Returned by
            :py:meth:`torch.utils.data.Dataset.__getitem__`.
        **kwargs : `dict`
            Additional keyword arguments passed to
            :py:func:`torch.nn.functional.pad`..

        Returns
        -------
        xb : :py:class:`torch.Tensor`
            Mini-batch of training input data.
        yb : :py:class:`torch.Tensor`
            Mini-batch of training labels.

        """
        # find longest time series among current batch
        max_len = max([x[0].shape[self.dim] for x in batch])

        # pad tensors to longest time series length
        batch = [(PadCollate.center_pad(x[0], size=max_len, dim=self.dim,
                                        **kwargs), x[1]) for x in batch]

        # stack training samples to mini-batch
        xb = torch.stack([x[0] for x in batch], dim=0)
        yb = torch.stack([x[1] for x in batch], dim=0)

        return xb, yb

    def __call__(self, batch):
        """Wrapper for :py:meth:`ai4ebv.core.dataset.PadCollate.pad_collate`.

        Parameters
        ----------
        batch : `list` [`tuple`]
            List of tuples (input data, labels). Returned by
            :py:meth:`torch.utils.data.Dataset.__getitem__`.

        Returns
        -------
        xb : :py:class:`torch.Tensor`
            Mini-batch of training input data.
        yb : :py:class:`torch.Tensor`
            Mini-batch of training labels.

        """
        return self.pad_collate(batch)

    @staticmethod
    def center_pad(tensor, size, dim, **kwargs):
        """Center-pad a tensor to ``size`` along ``dim``.

        Parameters
        ----------
        tensor : :py:class:`torch.Tensor`
            The tensor to pad.
        size : `int`
            The size to pad ``dim`` to.
        dim : `int`
            The dimension to pad along.
        **kwargs : `dict`
            Additional keyword arguments passed to
            :py:func:`torch.nn.functional.pad`.

        Returns
        -------
        ptensor : :py:class:`torch.Tensor`
            The padded tensor.

        """
        # amount of padding
        to_pad = size - tensor.shape[dim]

        # check whether amount of padding is even or uneven
        if to_pad % 2 > 0:
            pad = (np.ceil(to_pad / 2), np.floor(to_pad / 2))
        else:
            pad = 2 * (to_pad / 2,)

        # tuple defining amount of padding along each dimension
        padding = np.asarray(tensor.ndim * (0, 0))
        padding[(2 * dim)] = pad[0]
        padding[(2 * dim) + 1] = pad[1]

        return F.pad(tensor, tuple(padding[::-1]), **kwargs)
