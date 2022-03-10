"""Graphic settings."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# externals
import matplotlib.pyplot as plt

# plot font size configuration
SMALL = 16
MEDIUM = 18
BIG = 20

# controls default font size
plt.rc('font', size=MEDIUM)

# axes labels size
plt.rc('axes', titlesize=BIG, labelsize=MEDIUM)

# axes ticks size
plt.rc('xtick', labelsize=SMALL)
plt.rc('ytick', labelsize=SMALL)

# legend font size
plt.rc('legend', fontsize=MEDIUM)

# figure title size
plt.rc('figure', titlesize=BIG)
