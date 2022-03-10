"""PyTorch model and training configuration."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# locals
from ai4ebv.core.models import MLP, FCN, ANN

# the classifier
# MODEL = ANN
# MODEL = MLP
MODEL = FCN

# whether overwrite models
TORCH_OVERWRITE = False

# training batch size
BATCH_SIZE = 10000

# inference block size
TILE_SIZE = (256, 256)

# learning rate
LR = 0.001

# network training configuration
TRAIN_CONFIG = {
    'checkpoint_state': {},
    'epochs': 250,
    'save': True,
    'save_loaders': False,
    'early_stop': True,
    'patience': 25,
    'multi_gpu': True,
    'classification': True
    }
