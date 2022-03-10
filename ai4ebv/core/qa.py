"""Functions to handle 8-bit encoded quality assessment layers."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import logging
from operator import itemgetter

# externals
import numpy as np

# locals
from pysegcnn.core.utils import dec2bin

# module level logger
LOGGER = logging.getLogger(__name__)

# quality assessment parameters for the Harmonized Landsat8-Sentinel2 dataset
QUALITY_ASSESSMENT_PARAMETERS = [
    'Aerosol quality 1', 'Aerosol quality 2', 'Water', 'Snow and Ice',
    'Cloud shadow', 'Adjacent cloud', 'Cloud', 'Cirrus']


def quality_assessment_parameters(binary):
    """Convert the binary quality assessment number to the quality metadata.

    Parameters
    ----------
    binary : `str`
        The 8-bit quality assessment binary number.

    Returns
    -------
    qa : `dict` [`str`, `int`]
        A dictionary containing the quality metadata.

    """
    # quality assessment dictionary
    qa = {k: int(bit) for k, bit in zip(QUALITY_ASSESSMENT_PARAMETERS, binary)}
    return qa


# quality assessment layer: convert binary string to quality metadata integers
QUALITY_ASSESSMENT_MAPPING = {
    k: quality_assessment_parameters(dec2bin(k)) for k in range(256)}


def qa2binary(qa, parameters=['Cloud', 'Cirrus', 'Cloud shadow']):
    """Decode 8-bit encoded quality assessment layer to a binary mask.

    Parameters
    ----------
    qa : :py:class:`numpy.ndarray`, shape=(height, width)
        The quality assessment layer encoded as 8-bit unsigned integer.
    parameters : `list` [`str`], optional
        Parameters to mask. The default is ['Cloud', 'Cirrus', 'Cloud shadow'].

    Returns
    -------
    binary_qa : :py:class:`numpy.ndarray`, shape=(height, width)
        A binary mask, where values matching any of the flags specified in
        ``parameters`` are set to True.

    """
    # check if the specified parameters exist in the quality assessment layer
    if not all([p in QUALITY_ASSESSMENT_PARAMETERS for p in parameters]):
        LOGGER.warning('Specified parameters "{}" are not valid for the '
                       'quality assessment layer with parameters "{}".'
                       .format(', '.join(parameters),
                               ', '.join(QUALITY_ASSESSMENT_PARAMETERS)))
        LOGGER.info('Can not apply quality assessment mask.')
        return

    # get the qa-values that represent any of the values to mask
    qa_to_mask = [k for k, v in QUALITY_ASSESSMENT_MAPPING.items() if
                  any(*[itemgetter(*parameters)(v)])]

    # binary quality assessment layer: True, for values matching parameters
    #                                  False, otherwise
    return np.isin(qa, qa_to_mask)
