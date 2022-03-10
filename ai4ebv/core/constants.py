"""Constant parameters."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import re

# locals
from pysegcnn.core.constants import Landsat8, Sentinel2
from ai4ebv.core.legend import Legend

# the order of the bands of the Landsat-8 HLS scenes
L8BANDS = list(Landsat8.band_dict().values())
L8BANDS.remove('pan')
L8BANDS.append('qa')

# the order of the bands of the Sentinel-2 HLS scenes
S2BANDS = list(Sentinel2.band_dict().values())
S2BANDS.append('qa')

# create dictionaries mapping band names to subdataset numbers
L8SUBDATASETS = {k: '0{}'.format(v + 1) if v < 9 else '{}'.format(v + 1) for
                 v, k in enumerate(L8BANDS)}
S2SUBDATASETS = {k: '0{}'.format(v + 1) if v < 9 else '{}'.format(v + 1) for
                 v, k in enumerate(S2BANDS)}

# spectral bands to use for training: include the quality assessment band to
# mask clouds and cloud shadows
USE_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'qa']

# quality assessment flags to mask as NoData
QA_TO_MASK = ['Cloud', 'Cirrus', 'Cloud shadow']

# class labels: drop the NoData class
LABELS = Legend.label_dict()
LABELS.pop(Legend.NoData.id)

# original class labels
OLABELS = list(LABELS.keys())

# class names
CLASS_LABELS = [v['label'] for v in LABELS.values()]

# define patterns to parse tile name and year
HLS_TILE = re.compile('T[0-9]{2}[A-Z]{3}')
HLS_DATE = re.compile('[0-9]{4}[0-3][0-9][0-9]')

# pattern for MGRS tile names: UTM zone (1 - 60), Pos: (C - X, A - Z, A - V)
TILE_PATTERN = '_[1-6][0-9][C-X][A-Z][A-V].tif$'

# tiles covering the European Alps
ALPS_TILES = ('31TFJ', '31TFK', '31TFL', '31TGH', '31TGJ', '31TGK', '31TGL',
              '31TGM', '32TLP', '32TLQ', '32TLR', '32TLS', '32TLT', '32TMP',
              '32TMQ', '32TMR', '32TMS', '32TMT', '32TNR', '32TNS', '32TNT',
              '32TPR', '32TPS', '32TPT', '32TQR', '32TQS', '32TQT', '32UNU',
              '32UPU', '32UQU', '33TUL', '33TUM', '33TUN', '33TVL', '33TVM',
              '33TVN', '33TWM', '33TWN', '33TXN', '33UUP', '33UVP', '33UWP',
              '33UXP')

# tiles covering the province of Trentino-Südtirol
STT_TILES = ('32TPS', '32TPT', '32TQS', '32TQT')

SMALL_TILES = ('ST')

# tiles covering the Himalayas
HIMALAYAS_TILES = ('43REQ', '43RFP', '43RFQ', '43RGP', '43RGQ', '43SER',
                   '43SES', '43SFR', '43SFS', '43SGR', '43SGS', '43RGN',
                   '43SBA', '43SBV', '43SCA', '43SCS', '43SCT', '43SCU',
                   '43SCV', '43SDA', '43SDB', '43SDR', '43SDS', '43SDT',
                   '43SDU', '43SDV', '43SEA', '43SEB', '43SET', '43SEU',
                   '43SEV', '43SFT', '43SFU', '43SFV', '43SGU', '43SGV',
                   '43SGT', '44RKV', '44RLV', '44SKA', '44SKB', '44RLS',
                   '44RLT', '44RMS', '44RMT', '44RMU', '44RNR', '44RNS',
                   '44RNT', '44RNU', '44RPR', '44RPS', '44RPT', '44RPU',
                   '44RQR', '44RQS', '44RQT', '44RKT', '44RKU', '44RLU',
                   '44RMV', '44SKC', '44SKD', '44SKE', '44SLA', '44SLB',
                   '44SLC', '44SLD', '45RTK', '45RTL', '45RTM', '45RTN',
                   '45RUK', '45RUL', '45RUM', '45RVK', '45RVL', '45RVM',
                   '45RWK', '45RWL', '45RWM', '45RXK', '45RXL')

# tiles of the Himalayan study site
HIMALAYAS_SITE = ('45RVM', '45RVL', '45RUM', '45RUL')

# tiles covered by the local LISS benchmark dataset
LISS_TILES = ('32TPS', '32TQS', '32TPT', '32TQT')

# coordinate reference system EPSG codes for final mosaics
ALPS_CRS = 3035  # ETRS89-extended /  LAEA Europe
HIMALAYAS_CRS = 32645  # WGS 84 / UTM zone 45N
