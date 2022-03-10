"""The classification legend of the Hammond landforms."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import warnings
import logging
from collections.abc import Iterable

# externals
import numpy as np
from osgeo import gdal
from scipy.signal import convolve, fftconvolve
from scipy.ndimage import maximum_filter, minimum_filter

# locals
from pysegcnn.core.constants import Label, LabelMapping

# constants: Hammond slope, relief, and profile classes
HAMMOND_SLOPE_CLASSES = {0: 400., 20: 300., 50: 200., 80: 100.}
HAMMOND_RELIEF_CLASSES = {0: 10, 30: 20, 90: 30, 150: 40, 300: 50, 900: 60}
HAMMOND_PROFILE_LOWLANDS = {0: 0, 50: 2, 75: 1}
HAMMOND_PROFILE_UPLANDS = {0: 0, 50: 3, 75: 4}

# kernels for computation of slope after Horn (1981)
HORN_KERNEL_DX = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
HORN_KERNEL_DY = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])

# module level logger
LOGGER = logging.getLogger(__name__)


class HammondLandforms(Label):
    """The Hammond Landforms after `Karagulle et al. (2017)`_.

    .. _Karagulle et al. (2017):
        https://onlinelibrary.wiley.com/doi/full/10.1111/tgis.12265

    """
    High_hills = (140, 141, 142, 143, 144, 440, 441, 442), (123, 166, 75)
    Low_mountains = (150, 151, 152, 153, 154, 250, 251, 252, 253, 254, 350,
                     351, 352, 450, 451, 452), (206, 167, 112)
    High_mountains = (160, 161, 162, 163, 164, 260, 261, 262, 263, 264, 360,
                      361, 362, 460, 461, 462), (117, 81, 16)
    Scattered_moderate_hills = (230, 231, 232, 233, 234, 330, 331, 332, 430,
                                431, 432), (86, 172, 65)
    Scattered_high_hills = ((240, 241, 242, 243, 244, 340, 341, 342),
                            (122, 159, 66))
    Irregular_plains_with_low_hills = 320, (234, 253, 198)
    Irregular_plains_with_moderate_relief = ((321, 322, 323, 324),
                                             (229, 254, 195))
    Tablelands_with_moderate_relief = (333, 334, 433, 434), (194, 159, 217)
    Tablelands_with_considerable_relief = (343, 344, 443, 444), (172, 144, 167)
    Tablelands_with_high_relief = (353, 354, 453, 454), (166, 123, 133)
    Tablelands_with_very_high_relief = (363, 364, 463, 464), (138, 108, 105)
    Flat_or_nearly_flat_plains = ((410, 411, 412, 413, 414, 420),
                                  (254, 255, 189))
    Smooth_plains_with_some_local_relief = (421, 422, 423, 424), (250, 251, 25)


# Hammond landform classes after Karagulle et al. (2017)
HAMMOND_LANDFORM_CLASSES = []
for k in HammondLandforms.label_dict().keys():
    (HAMMOND_LANDFORM_CLASSES.extend(k) if isinstance(k, Iterable) else
     HAMMOND_LANDFORM_CLASSES.append(k))
HAMMOND_LANDFORM_CLASSES = np.asarray(sorted(HAMMOND_LANDFORM_CLASSES))


class SayreLandforms(Label):
    """The aggregated Hammond Landforms applied by `Sayre et al. (2020)`_.

    .. _Sayre et al. (2020):
        https://www.sciencedirect.com/science/article/pii/S2351989419307231

    """
    Mountains = 1, (137, 112, 68)
    Hills = 2, (112, 168, 0)
    Plains = 3, (255, 115, 223)
    Tablelands = 4, (255, 255, 190)
    NoData = 255, (0, 0, 0)

# landform classes after Sayre et al. (2020)
SAYRE_LANDFORM_CLASSES = np.asarray(
    sorted(list(SayreLandforms.label_dict().keys())))[0:-1]

# aggregation of Hammond Landforms after Sayre et al. (2020)
LandformAggregation = LabelMapping({

    # Mountains
    HammondLandforms.Low_mountains.id: SayreLandforms.Mountains.id,
    HammondLandforms.High_mountains.id: SayreLandforms.Mountains.id,

    # Hills
    HammondLandforms.High_hills.id: SayreLandforms.Hills.id,
    HammondLandforms.Scattered_moderate_hills.id: SayreLandforms.Hills.id,
    HammondLandforms.Scattered_high_hills.id: SayreLandforms.Hills.id,
    HammondLandforms.Irregular_plains_with_low_hills.id:
        SayreLandforms.Hills.id,
    HammondLandforms.Irregular_plains_with_moderate_relief.id:
        SayreLandforms.Hills.id,

    # Plains
    HammondLandforms.Flat_or_nearly_flat_plains.id: SayreLandforms.Plains.id,
    HammondLandforms.Smooth_plains_with_some_local_relief.id:
        SayreLandforms.Plains.id,

    # Tablelands
    HammondLandforms.Tablelands_with_moderate_relief.id:
        SayreLandforms.Tablelands.id,
    HammondLandforms.Tablelands_with_considerable_relief.id:
        SayreLandforms.Tablelands.id,
    HammondLandforms.Tablelands_with_high_relief.id:
        SayreLandforms.Tablelands.id,
    HammondLandforms.Tablelands_with_very_high_relief.id:
        SayreLandforms.Tablelands.id,

    })


# generic class for neighborhood analysis windows
class NeighborhoodAnalysisWindow(object):
    """Generic class to define a neighborhood analysis window (NAW)."""

    # supported convolutional modes
    modes = ['sum', 'mean']

    def __init__(self, mode='sum'):
        """Initialize the neighbhorhood analysis window.


        Parameters
        ----------
        mode : `str`, optional
            Statistic to compute within the NAW. The default is `'sum'`.

        Raises
        ------
        ValueError
            If the statistic ``mode`` is not supported.

        """
        # check if mode is supported
        if mode not in self.modes:
            raise ValueError('Mode "{}" not supported. Available modes are {}.'
                             .format(mode, ','.join(self.modes)))
        self.mode = mode

    @property
    def neighborhood(self):
        """A boolean mask defining the shape of the NAW.

        Raises
        ------
        NotImplementedError
            This property depends on the type of the neighborhood and needs to
            be implemented when inheriting this class.

        Returns
        -------
        neighborhood : :py:class:`numpy.ndarray`
            The neighorhood analysis window described as boolean mask.

        """
        raise NotImplementedError('Define valid cells for this neighborhood.')

    @property
    def kernel(self):
        """The convolutional kernel derived from the NAW.

        This kernel is used to compute the statistics within the NAW around
        each pixel of the digital elevation model.


        Returns
        -------
        kernel : :py:class:`numpy.ndarray`
            The convolutional kernel to compute the statistics within the NAW.

        """
        return (self.neighborhood if self.mode == 'sum' else
                self.neighborhood / np.count_nonzero(self.neighborhood))


# rectangular NAWs
class RectangularNAW(NeighborhoodAnalysisWindow):
    """Rectangular neighborhood analysis window."""

    def __init__(self, mode='sum', size=(3, 3)):
        """Initialize.

        Parameters
        ----------
        mode : `str`, optional
            Statistic to compute within the NAW. The default is `'sum'`.
        size : `tuple` [`int`, `int`], optional
            The size of the rectangular NAW in pixels. The default is `(3, 3)`.

        """
        super().__init__(mode)

        # size of rectangular neighborhood
        self.height, self.width = size

    @property
    def neighborhood(self):
        """Boolean mask of a rectangular NAW."""
        return np.ones(shape=(self.height, self.width))


# ciruclar NAWs
class CircularNAW(NeighborhoodAnalysisWindow):
    """Circular neighborhood analysis window."""

    def __init__(self, mode='sum', radius=3):
        """Initialize.

        Parameters
        ----------
        mode : `str`, optional
            Statistic to compute within the NAW. The default is `'sum'`.
        radius : `int` or `float`
            The radius of the circular NAW in pixels. The default is `3`.

        """
        super().__init__(mode)

        # radius of circular neighborhood
        self.radius = radius

    @property
    def neighborhood(self):
        """Boolean mask of a circular NAW with defined radius."""
        # mask of pixels within circular neighborhood
        y, x = np.ogrid[-self.radius:self.radius+1, -self.radius:self.radius+1]
        return (x ** 2 + y ** 2 <= self.radius ** 2).astype(np.float32)


def fftconvolve_and_pad(array, kernel):
    """Wrapper for valid fast Fourier transform convolution and padding.

    Parameters
    ----------
    array : :py:class:`numpy.ndarray`, shape=(n, ...)
        Array to convolve.
    kernel : :py:class:`numpy.ndarray`, shape=(n, ...)
        Convolutional kernel.

    Returns
    -------
    fft_convolved : :py:class:`numpy.ndarray`, shape=(n, ...)
        Valid convolution of ``array`` with ``kernel``.

    """
    return np.pad(fftconvolve(array, kernel, mode='valid'),
                  int(kernel.shape[0] / 2), mode='constant',
                  constant_values=np.nan)

def reclassify(array, scheme):
    """Reclassify an array based on a defined classification scheme.


    Parameters
    ----------
    array : :py:class:`numpy.ndarray`, shape=(n, ...)
        Array to reclassify.
    scheme : `dict`
        Classification scheme. A dictionary with keys describing the class
        boundaries and values the corresponding classes.

    Returns
    -------
    reclassified : :py:class:`numpy.ndarray`, shape=(n, ...)
        Reclassified array.

    """
    # reclassify based on lookup scheme
    reclassified = np.ones(array.shape) * np.nan
    class_boundaries = list(scheme.keys())
    for i, c in enumerate(class_boundaries):

        # condition to be met for current class boundary
        if i == 0:
            # condition at the lower boundary
            condition = (array >= c) & (array <= class_boundaries[i + 1])
        elif i + 1 == len(class_boundaries):
            # condition at the upper boundary
            condition = (array > c)
        else:
            # conditions between lower and upper class boundaries
            condition = (array > c) & (array <= class_boundaries[i + 1])

        # reclassify to current class
        reclassified[condition] = scheme[c]

    # check if values are correctly reclassified
    assert np.isin(np.unique(reclassified[~np.isnan(reclassified)]),
                   np.asarray(list(scheme.values()))).all()

    return reclassified


def dem_slope_horn(dem, cell_size, units='percent'):
    """Compute slope of a digital elevation model after `Horn (1981)`_.

    Adapted from the implementation of `"Slope"`_ in ArcGIS.

    Note
    ----
    The ``dem`` must be defined in a *metric* coordinate system and NoData
    values must be set to ``np.nan`` in ``dem`` to avoid artefacts. Slope
    values for cells near NoData values are set to ``np.nan``.

    Parameters
    ----------
    dem : :py:class:`numpy.ndarray`, shape=(height, width)
        The digital elevation model.
    cell_size : `tuple` or :py:class:`numpy.ndarray`, shape=(dx, dy)
        The size of a pixel in meters.
    units : `str`, optional
        Whether to compute slope in percent or degrees. The default is
        `'percent'`.

    Returns
    -------
    slope : :py:class:`numpy.ndarray`, shape=(height, width)
        Slope of the digital elevation model.

    .. _Horn (1981):
        https://ieeexplore.ieee.org/document/1456186

    .. _"Slope":
        https://desktop.arcgis.com/de/arcmap/latest/tools/spatial-analyst-toolbox/how-slope-works.htm

    """
    # compute gradients: dz/dx and dz/dy
    dz_dx = convolve(dem, HORN_KERNEL_DX / (8 * cell_size[0]), mode='same',
                     method='direct')
    dz_dy = convolve(dem, HORN_KERNEL_DY / (8 * cell_size[1]), mode='same',
                     method='direct')
    rise_run = np.sqrt(dz_dx ** 2 + dz_dy ** 2)

    # return slope in percent or degrees
    return (rise_run * 100 if units == 'percent' else
            np.arctan(rise_run) * 180 / np.pi)


# function to calculate Hammond Landforms after Karagulle et al. (2017)
def hammond_landforms(dem,
                      slope_naw=CircularNAW(mode='mean', radius=33),
                      relief_naw=CircularNAW(mode='sum', radius=33),
                      profile_naw=CircularNAW(mode='sum', radius=33),
                      no_data=9999):
    """Compute Hammond Landforms after `Karagulle et al. (2017)`_.

    .. _Karagulle et al. (2017):
        https://onlinelibrary.wiley.com/doi/full/10.1111/tgis.12265

    Note
    ----
    The ``dem`` must be defined in a *metric* coordinate system.

    Parameters
    ----------
    dem : `str` or :py:class:`pathlib.Path`
        The digital elevation model.
    slope_naw : :py:class:`ai4ebv.core.landforms.NeighborhoodAnalysisWindow`
        Slope neighborhood analysis window. The default is
        `CircularNAW(mode='mean', radius=33)`.
    relief_naw : :py:class:`ai4ebv.core.landforms.NeighborhoodAnalysisWindow`
        Relief neighborhood analysis window. The default is
        `CircularNAW(mode='mean', radius=33)`.
    profile_naw : :py:class:`ai4ebv.core.landforms.NeighborhoodAnalysisWindow`
        Profile neighborhood analysis window. The default is
        `CircularNAW(mode='mean', radius=33)`.
    no_data : `int`, optional
        Class label to assign to NoData values. ``no_data`` should be a
        positive integer in the UInt16 range. The default is `9999`.

    Returns
    -------
    hammond_landforms : :py:class:`numpy.ndarray`
        The Hammond landforms as UInt16 data type.

    """
    # read digital elevation model to numpy.ndarray
    LOGGER.info('Reading digital elevation model: {}'.format(str(dem)))
    elevation = gdal.Open(str(dem))
    cell_size = np.abs([elevation.GetGeoTransform()[1],
                        elevation.GetGeoTransform()[-1]])
    srcNoData = elevation.GetRasterBand(1).GetNoDataValue()
    elevation = elevation.GetRasterBand(1).ReadAsArray().astype(np.float32)

    # replace srcNoData value by np.nan
    elevation[elevation == srcNoData] = np.nan

    # -------------------------------------------------------------------------
    # A: Hammond Slope Classes ------------------------------------------------
    # -------------------------------------------------------------------------
    LOGGER.info('Computing Hammond slope classes ...')

    # A1: calculate percent slope
    if np.isnan(elevation).any():
        # calculate slope with custom function to properly handle NoData values
        a1 = dem_slope_horn(elevation, cell_size)
    else:
        # calculate slope with gdal's version of Horn's (1981) algorithm
        # NOTE: this is buggy for DEMs containing NoData values
        a1 = gdal.DEMProcessing(
            '', str(dem), 'slope', format='MEM', computeEdges=False,
            alg='Horn', slopeFormat='percent').ReadAsArray().astype(np.float32)

    # A2: reclassify percent slope to gentle slope
    # percent slope < 8%: 0
    # percent slope > 8%: 1
    a2 = np.where(a1 <= 8, 0, 1).astype(np.float32)

    # A3: percent of gentle slope in slope NAW
    a3 = fftconvolve_and_pad(a2, slope_naw.kernel) * 100
    a3[np.isnan(a1)] = np.nan  # mask NoData values
    a3 = a3.clip(min=0, max=100)  # clip values to valid range

    # A4: Hammond slope classes
    hammond_slope = reclassify(a3, HAMMOND_SLOPE_CLASSES)

    # -------------------------------------------------------------------------
    # B: Hammond Relief Classes -----------------------------------------------
    # -------------------------------------------------------------------------
    LOGGER.info('Computing Hammond relief classes ...')

    # B1: calculate maximum elevation within relief NAW
    b1 = maximum_filter(elevation, footprint=relief_naw.neighborhood,
                        mode='constant', cval=np.nan).astype(np.float32)

    # B2: calculate minimum elevation within relief NAW
    b2 = minimum_filter(elevation, footprint=relief_naw.neighborhood,
                        mode='constant', cval=np.nan).astype(np.float32)

    # B3: calculate local relief
    b3 = (b1 - b2).clip(min=0)

    # B4: Hammond relief classes
    hammond_relief = reclassify(b3, HAMMOND_RELIEF_CLASSES)

    # -------------------------------------------------------------------------
    # C: Hammond Profile Classes ----------------------------------------------
    # -------------------------------------------------------------------------
    LOGGER.info('Computing Hammond profile classes ...')

    # C1: local point of distinction between lowlands and uplands
    c1 = b3 / 2

    # C2: elevation of local point of distinction between lowlands and uplands
    c2 = c1 + b2

    # C3: surface of lowlands vs. uplands
    c3 = elevation - c2

    # C4/C5/C15: classify to lowlands and uplands
    c4_lowlands = (c3 <= 0).astype(np.float32)
    c4_uplands = (c3 > 0).astype(np.float32)

    # C6: gentle slope in lowlands
    c6 = c4_lowlands * a2

    # C7: sum of gentle slopes in lowlands
    c7 = fftconvolve_and_pad(c6, profile_naw.kernel)

    # C8: sum of gentle slopes
    c8 = fftconvolve_and_pad(a2, profile_naw.kernel)

    # C9: percent of gentle slopes in lowlands
    with warnings.catch_warnings():
        # do not log warnings due to 0-division to
        warnings.simplefilter('ignore', category=RuntimeWarning)
        c9 = c7 / c8

    # C10-11: replace NoData values by zero
    c10 = np.nan_to_num(c9, copy=True, nan=0, posinf=0, neginf=0)

    # C12: isolate gentle slopes in lowlands
    c12 = c10 * c4_lowlands

    # C13: precent of gentle slopes in lowlands
    c13 = fftconvolve_and_pad(c12, slope_naw.kernel) * 100
    c13[np.isnan(a1)] = np.nan  # mask NoData values
    c13 = c13.clip(min=0, max=100)  # clip values to valid range

    # C14: profile in lowland areas
    c14 = reclassify(c13, HAMMOND_PROFILE_LOWLANDS)

    # C16: gentle slope in uplands
    c16 = c4_uplands * a2

    # C17: sum of gentle slopes in uplands
    c17 = fftconvolve_and_pad(c16, profile_naw.kernel)

    # C18: percent of gentle slopes in uplands
    with warnings.catch_warnings():
        # do not log warnings due to 0-division to
        warnings.simplefilter('ignore', category=RuntimeWarning)
        c18 = c17 / c8

    # C19-C20: replace NoData values by zero
    c19 = np.nan_to_num(c18, copy=True, nan=0, posinf=0, neginf=0)

    # C21: isolate gentle slopes in uplands
    c21 = c19 * c4_uplands

    # C22: precent of gentle slopes in uplands
    c22 = fftconvolve_and_pad(c21, slope_naw.kernel) * 100
    c22[np.isnan(a1)] = np.nan  # mask NoData values
    c22 = c22.clip(min=0, max=100)  # clip values to valid range

    # C23: profile in upland areas
    c23 = reclassify(c22, HAMMOND_PROFILE_UPLANDS)

    # C24: Hammond profile classes
    hammond_profile = c14 + c23

    # -------------------------------------------------------------------------
    # D: Hammond Landforms ----------------------------------------------------
    # -------------------------------------------------------------------------
    LOGGER.info('Computing Hammond landforms ...')

    # compute hammond landforms
    hammond_landforms = hammond_slope + hammond_relief + hammond_profile

    # replace NoData value and convert to UInt16
    hammond_landforms[np.isnan(hammond_landforms)] = no_data
    hammond_landforms = hammond_landforms.astype(np.uint16)

    return hammond_landforms
