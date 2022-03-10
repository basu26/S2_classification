"""The working legend of the land cover classification."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# locals
from pysegcnn.core.constants import Label, LabelMapping
from ai4ebv.core.landcover import (EsaCciLc, CorineLandCover, HumboldtLC,
                                   WteLandCover, LISS)


class Legend(Label):
    """The land cover legend to classify."""
    
    NoData = 255, (0, 0, 0)
    Artificial_surfaces = 120, (230, 0, 0)
    Farmland = 210, (255, 234, 190)
    Vineyards = 221, (255, 211, 127)
    Fruit_plantations = 222, (230, 152, 0)
    Intensively_used_meadows_and_pastures = 231, (76, 230, 0)
    Mixed_agricultural_areas = 250, (115, 76, 0)
    Forest = 315, (38, 115, 0)
    Extensively_used_pastures_and_natural_meadows = 321, (233, 255, 190)
    Krummholz = 322, (168, 168, 0)
    Dwarf_shrubs = 323, (160, 230, 77)
    Grassland_with_trees = 324, (173, 199, 46)
    Bare_rock = 332, (225, 225, 225)
    Sparse_vegetation = 333, (104, 104, 104)
    Glaciers = 430, (255, 190, 232)
    Wetlands = 510, (190, 232, 255)
    Water_bodies = 520, (0, 112, 255)


#    Crops = 1, (255, 255, 0)
#    Orchards = 2, (242, 166, 77)
#    Vineyards = 3, (230, 128, 0)
#    Shrubland = 4, (150, 100, 0)
#    Forest_broadleaved = 5, (128, 255, 0)
#    Forest_coniferous = 6, (0, 166, 0)
#    Grassland = 7, (255, 180, 50)
#    Settlement = 8, (195, 20, 0)
#    Bare_areas = 9, (255, 235, 175)
#    Water = 10, (190, 232, 255)
#    Snow_and_Ice = 11, (255, 255, 255)
#    NoData = 255, (0, 0, 0)


