"""The classification legends of the different land cover datasets."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# locals
from pysegcnn.core.constants import Label, LabelMapping


class WteLandCover(Label):
    """The World Terrestrial Ecosystems Land Cover legend."""

    Cropland = 1, (255, 255, 0)
    Shrubland = 2, (150, 100, 0)
    Forest = 3, (0, 160, 0)
    Grassland = 4, (255, 180, 50)
    Settlement = 5, (195, 20, 0)
    Bare_areas = 6, (255, 235, 175)
    Water = 7, (190, 232, 255)
    Snow_and_Ice = 8, (106, 255, 255)
    NoData = 255, (0, 0, 0)


# Class mapping from WTE Land Cover to itself
Wte2Wte = LabelMapping({k: k for k in WteLandCover.label_dict().keys()})


class EsaCciLc(Label):
    """The `ESA CCI Land Cover`_ legend.

    .. _ESA CCI Land Cover:
        http://www.esa-landcover-cci.org/

    """
    Rainfed_cropland = 10, (255, 255, 100)
    Rainfed_herbaceous_covered_cropland = 11, (255, 255, 100)
    Rainfed_tree_or_shrub_covered_cropland = 12, (255, 255, 0)
    Irrigated_cropland = 20, (170, 240, 240)
    Mosaic_cropland_and_natural_vegetation = 30, (220, 240, 100)
    Mosaic_natural_vegetation_and_cropland = 40, (200, 200, 100)
    Tree_cover_broadleaved_evergreen_closed_to_open = 50, (0, 100, 0)
    Tree_cover_broadleaved_deciduous_closed_to_open = 60, (0, 160, 0)
    Tree_cover_broadleaved_deciduous_closed = 61, (0, 160, 0)
    Tree_cover_broadleaved_deciduous_open = 62, (170, 200, 0)
    Tree_cover_needleleaved_evergreen_closed_to_open = 70, (0, 60, 0)
    Tree_cover_needleleaved_evergreen_closed = 71, (0, 60, 0)
    Tree_cover_needleleaved_evergreen_open = 72, (0, 80, 0)
    Tree_cover_needleleaved_deciduous_closed_to_open = 80, (40, 80, 0)
    Tree_cover_needleleaved_deciduous_closed = 81, (40, 80, 0)
    Tree_cover_needleleaved_deciduous_open = 82, (40, 100, 0)
    Tree_cover_mixed_leaf_type = 90, (120, 130, 0)
    Mosaic_tree_and_shrub_and_herbaceous_cover = 100, (140, 160, 0)
    Mosaic_herbaceous_cover_and_tree_and_shrub = 110, (190, 150, 0)
    Shrubland = 120, (150, 100, 0)
    Evergreen_shrubland = 121, (120, 75, 0)
    Deciduous_shrubland = 122, (150, 100, 0)
    Grassland = 130, (255, 180, 50)
    Lichens_and_mosses = 140, (255, 220, 210)
    Sparse_vegetation = 150, (255, 235, 175)
    Sparse_tree = 151, (255, 200, 100)
    Sparse_shrub = 152, (255, 210, 120)
    Sparse_herbaceous_cover = 153, (255, 235, 175)
    Tree_cover_flooded_fresh_or_brackish_water = 160, (0, 120, 90)
    Tree_cover_flooded_saline_water = 170, (0, 150, 120)
    Shrub_or_herbaceous_cover_flooded = 180, (0, 220, 130)
    Urban_areas = 190, (195, 20, 0)
    Bare_areas = 200, (255, 245, 215)
    Consolidated_bare_areas = 201, (220, 220, 220)
    Unconsolidated_bare_areas = 202, (255, 245, 215)
    Water_bodies = 210, (0, 70, 200)
    Permanent_snow_and_ice = 220, (255, 255, 255)
    NoData = 255, (0, 0, 0)


# Class label mapping from ESA CCI Land Cover to WTE Land Cover
EsaCciLc2Wte = LabelMapping({

    # Cropland
    EsaCciLc.Rainfed_cropland.id: WteLandCover.Cropland.id,
    EsaCciLc.Rainfed_herbaceous_covered_cropland.id: WteLandCover.Cropland.id,
    EsaCciLc.Rainfed_tree_or_shrub_covered_cropland.id:
        WteLandCover.Cropland.id,
    EsaCciLc.Irrigated_cropland.id: WteLandCover.Cropland.id,
    EsaCciLc.Mosaic_cropland_and_natural_vegetation.id:
        WteLandCover.Cropland.id,
    EsaCciLc.Mosaic_natural_vegetation_and_cropland.id:
        WteLandCover.Cropland.id,

    # Forest
    EsaCciLc.Tree_cover_broadleaved_evergreen_closed_to_open.id:
        WteLandCover.Forest.id,
    EsaCciLc.Tree_cover_broadleaved_deciduous_closed_to_open.id:
        WteLandCover.Forest.id,
    EsaCciLc.Tree_cover_broadleaved_deciduous_closed.id:
        WteLandCover.Forest.id,
    EsaCciLc.Tree_cover_broadleaved_deciduous_open.id: WteLandCover.Forest.id,
    EsaCciLc.Tree_cover_needleleaved_evergreen_closed_to_open.id:
        WteLandCover.Forest.id,
    EsaCciLc.Tree_cover_needleleaved_evergreen_closed.id:
        WteLandCover.Forest.id,
    EsaCciLc.Tree_cover_needleleaved_evergreen_open.id: WteLandCover.Forest.id,
    EsaCciLc.Tree_cover_needleleaved_deciduous_closed_to_open.id:
        WteLandCover.Forest.id,
    EsaCciLc.Tree_cover_needleleaved_deciduous_closed.id:
        WteLandCover.Forest.id,
    EsaCciLc.Tree_cover_needleleaved_deciduous_open.id: WteLandCover.Forest.id,
    EsaCciLc.Tree_cover_mixed_leaf_type.id: WteLandCover.Forest.id,
    EsaCciLc.Tree_cover_flooded_fresh_or_brackish_water.id:
        WteLandCover.Forest.id,
    EsaCciLc.Tree_cover_flooded_saline_water.id: WteLandCover.Forest.id,
    EsaCciLc.Mosaic_tree_and_shrub_and_herbaceous_cover.id:
        WteLandCover.Forest.id,

    # Shrublands
    EsaCciLc.Mosaic_herbaceous_cover_and_tree_and_shrub.id:
        WteLandCover.Shrubland.id,
    EsaCciLc.Shrubland.id: WteLandCover.Shrubland.id,
    EsaCciLc.Evergreen_shrubland.id: WteLandCover.Shrubland.id,
    EsaCciLc.Deciduous_shrubland.id: WteLandCover.Shrubland.id,
    EsaCciLc.Shrub_or_herbaceous_cover_flooded.id: WteLandCover.Shrubland.id,

    # Grasslands
    EsaCciLc.Grassland.id: WteLandCover.Grassland.id,

    # Bare areas
    EsaCciLc.Lichens_and_mosses.id: WteLandCover.Bare_areas.id,
    EsaCciLc.Sparse_vegetation.id: WteLandCover.Bare_areas.id,
    EsaCciLc.Sparse_tree.id: WteLandCover.Bare_areas.id,
    EsaCciLc.Sparse_shrub.id: WteLandCover.Bare_areas.id,
    EsaCciLc.Sparse_herbaceous_cover.id: WteLandCover.Bare_areas.id,
    EsaCciLc.Bare_areas.id: WteLandCover.Bare_areas.id,
    EsaCciLc.Consolidated_bare_areas.id: WteLandCover.Bare_areas.id,
    EsaCciLc.Unconsolidated_bare_areas.id: WteLandCover.Bare_areas.id,

    # Settlements
    EsaCciLc.Urban_areas.id: WteLandCover.Settlement.id,

    # Water
    EsaCciLc.Water_bodies.id: WteLandCover.Water.id,

    # Permantent snow and ice
    EsaCciLc.Permanent_snow_and_ice.id: WteLandCover.Snow_and_Ice.id,

    # NoData
    EsaCciLc.NoData.id: WteLandCover.NoData.id

    })


class CorineLandCover(Label):
    """The `Copernicus Corine Land Cover`_ legend.

    .. _Copernicus Corine Land Cover:
        https://land.copernicus.eu/eagle/files/eagle-related-projects/pt_clc-conversion-to-fao-lccs3_dec2010

    """

    Continuous_urban_fabric = 111, (230, 0, 77)
    Discontinuous_urban_fabric = 112, (255, 0, 0)
    Industrial_or_commercial_units = 121, (204, 77, 242)
    Road_and_rail_networks_and_associated_land = 122, (204, 0, 0)
    Port_areas = 123, (230, 204, 240)
    Airports = 124, (230, 4, 230)
    Mineral_extraction_sites = 131, (166, 0, 204)
    Dump_sites = 132, (166, 77, 0)
    Construction_sites = 133, (255, 77, 255)
    Green_urban_areas = 141, (255, 166, 255)
    Sport_and_leisure_facilities = 142, (255, 230, 255)
    Non_irrigated_arable_land = 211, (255, 255, 168)
    Permanently_irrigated_land = 212, (255, 255, 0)
    Rice_fields = 213, (230, 230, 0)
    Vineyards = 221, (230, 128, 0)
    Fruit_trees_and_berry_plantations = 222, (242, 166, 77)
    Olive_groves = 223, (230, 166, 0)
    Pastures = 231, (230, 230, 77)
    Annual_crops_associated_with_permanent_crops = 241, (255, 230, 166)
    Complex_cultivation_patterns = 242, (255, 230, 77)
    Land_principally_occupated_by_agriculture = 243, (230, 204, 77)
    Agro_forestry_areas = 244, (242, 204, 166)
    Broad_leaved_forest = 311, (128, 255, 0)
    Coniferous_forest = 312, (0, 166, 0)
    Mixed_forest = 313, (77, 255, 0)
    Natural_grasslands = 321, (204, 242, 77)
    Moors_and_heathland = 322, (166, 255, 128)
    Sclerophyllous_vegetation = 323, (166, 230, 77)
    Transitional_woodland_shrub = 324, (166, 242, 0)
    Beaches_dunes_sands = 331, (230, 230, 230)
    Bare_rocks = 332, (204, 204, 204)
    Sparsely_vegetated_areas = 333, (204, 255, 204)
    Burnt_areas = 334, (0, 0, 0)
    Glaciers_and_perpetual_snow = 335, (166, 230, 204)
    Inland_marshes = 411, (166, 166, 255)
    Peat_bogs = 412, (77, 77, 255)
    Coastal_salt_marshes = 421, (204, 204, 255)
    Salines = 422, (230, 230, 255)
    Intertidal_flats = 423, (166, 166, 230)
    Water_courses = 511, (0, 204, 242)
    Water_bodies = 512, (128, 242, 230)
    Coastal_lagoons = 521, (0, 255, 166)
    Estuaries = 522, (166, 255, 230)
    Sea_and_ocean = 523, (230, 242, 255)
    NoData = 255, (0, 0, 0)


# Class label mapping from CORINE Land Cover to WTE Land Cover
Corine2Wte = LabelMapping({

    # Class 1: Artificial areas
    CorineLandCover.Continuous_urban_fabric.id: WteLandCover.Settlement.id,
    CorineLandCover.Discontinuous_urban_fabric.id: WteLandCover.Settlement.id,
    CorineLandCover.Industrial_or_commercial_units.id: (
        WteLandCover.Settlement.id),
    CorineLandCover.Road_and_rail_networks_and_associated_land.id: (
        WteLandCover.Settlement.id),
    CorineLandCover.Port_areas.id: WteLandCover.Settlement.id,
    CorineLandCover.Airports.id: WteLandCover.Settlement.id,
    CorineLandCover.Mineral_extraction_sites.id: WteLandCover.Settlement.id,
    CorineLandCover.Dump_sites.id: WteLandCover.Settlement.id,
    CorineLandCover.Construction_sites.id: WteLandCover.Settlement.id,
    CorineLandCover.Green_urban_areas.id: WteLandCover.Settlement.id,
    CorineLandCover.Sport_and_leisure_facilities.id: (
        WteLandCover.NoData.id),

    # Class 2: Agricultural areas
    CorineLandCover.Non_irrigated_arable_land.id: WteLandCover.Cropland.id,
    CorineLandCover.Permanently_irrigated_land.id: WteLandCover.Cropland.id,
    CorineLandCover.Rice_fields.id: WteLandCover.Cropland.id,
    CorineLandCover.Vineyards.id: WteLandCover.Cropland.id,
    CorineLandCover.Fruit_trees_and_berry_plantations.id: (
        WteLandCover.Cropland.id),
    CorineLandCover.Olive_groves.id: WteLandCover.Cropland.id,
    CorineLandCover.Pastures.id: WteLandCover.Grassland.id,
    CorineLandCover.Annual_crops_associated_with_permanent_crops.id: (
        WteLandCover.Cropland.id),
    CorineLandCover.Complex_cultivation_patterns.id: WteLandCover.Cropland.id,
    CorineLandCover.Land_principally_occupated_by_agriculture.id: (
        WteLandCover.Cropland.id),
    CorineLandCover.Agro_forestry_areas.id: WteLandCover.Forest.id,

    # Class 3: Forest and semi-natural areas
    CorineLandCover.Broad_leaved_forest.id: WteLandCover.Forest.id,
    CorineLandCover.Coniferous_forest.id: WteLandCover.Forest.id,
    CorineLandCover.Mixed_forest.id: WteLandCover.Forest.id,
    CorineLandCover.Natural_grasslands.id: WteLandCover.Grassland.id,
    CorineLandCover.Moors_and_heathland.id: WteLandCover.Shrubland.id,
    CorineLandCover.Sclerophyllous_vegetation.id: WteLandCover.Shrubland.id,
    CorineLandCover.Transitional_woodland_shrub.id: WteLandCover.Forest.id,
    CorineLandCover.Beaches_dunes_sands.id: WteLandCover.Bare_areas.id,
    CorineLandCover.Bare_rocks.id: WteLandCover.Bare_areas.id,
    CorineLandCover.Sparsely_vegetated_areas.id: WteLandCover.Bare_areas.id,
    CorineLandCover.Burnt_areas.id: WteLandCover.Bare_areas.id,
    CorineLandCover.Glaciers_and_perpetual_snow.id: (
        WteLandCover.Snow_and_Ice.id),

    # Class 4: Wetlands
    CorineLandCover.Inland_marshes.id: WteLandCover.NoData.id,
    CorineLandCover.Peat_bogs.id: WteLandCover.NoData.id,
    CorineLandCover.Coastal_salt_marshes.id: WteLandCover.NoData.id,
    CorineLandCover.Salines.id: WteLandCover.NoData.id,
    CorineLandCover.Intertidal_flats.id: WteLandCover.NoData.id,

    # Class 5: Water bodies
    CorineLandCover.Water_courses.id: WteLandCover.Water.id,
    CorineLandCover.Water_bodies.id: WteLandCover.Water.id,
    CorineLandCover.Coastal_lagoons.id: WteLandCover.Water.id,
    CorineLandCover.Estuaries.id: WteLandCover.Water.id,
    CorineLandCover.Sea_and_ocean.id: WteLandCover.Water.id,

    # Class 6: No data
    CorineLandCover.NoData.id: WteLandCover.NoData.id

})


class CopernicusGlobalLandCover(Label):
    """The `Copernicus Global Land Cover`_ legend.

    .. _Copernicus Global Land Cover:
        https://land.copernicus.eu/global/products/lc

    """

    # Forest
    Closed_evergreen_needleleaf_forest = 111, (88, 72, 31)
    Closed_deciduous_needleleaf_forest = 113, (112, 102, 62)
    Closed_evergreen_broadleaf_forest = 112, (0, 153, 0)
    Closed_deciduous_broadleaf_forest = 114, (0, 204, 0)
    Closed_mixed_forest = 115, (78, 117, 31)
    Closed_unknown_forest = 116, (0, 120, 0)
    Open_evergreen_needleleaf_forest = 121, (102, 96, 0)
    Open_deciduous_needleleaf_forest = 123, (141, 116, 0)
    Open_evergreen_broadleaf_forest = 122, (141, 180, 0)
    Open_deciduous_broadleaf_forest = 124, (160, 220, 0)
    Open_mixed_forest = 125, (146, 153, 0)
    Open_unknown_forest = 126, (100, 140, 0)

    # Shrublands and Grasslands
    Shrubs = 20, (255, 187, 34)
    Herbaceous_vegetation = 30, (255, 255, 76)

    # Wetlands
    Herbaceous_wetland = 90, (0, 150, 160)

    # Bare areas
    Bare_sparse_vegetation = 60, (180, 180, 180)
    Moss_and_lichen = 100, (250, 230, 160)

    # Croplands
    Cultivated_and_managed_vegetation = 40, (240, 140, 255)

    # Urban and built-up areas
    Urban_and_built_up = 50, (250, 0, 0)

    # Snow and ice
    Snow_and_ice = 70, (240, 240, 240)

    # Water bodies
    Water_bodies = 80, (0, 50, 200)

    # Open sea
    Open_sea = 200, (0, 0, 128)

    # Missing input data
    NoData = 255, (40, 40, 40)


# Class label mapping from Copernicus Global Land Cover to WTE Land Cover
Copernicus2Wte = LabelMapping({

    # Shrublands
    CopernicusGlobalLandCover.Shrubs.id: WteLandCover.Shrubland.id,

    # Grassland
    CopernicusGlobalLandCover.Herbaceous_vegetation.id: (
        WteLandCover.Grassland.id),

    # Cropland
    CopernicusGlobalLandCover.Cultivated_and_managed_vegetation.id: (
        WteLandCover.Cropland.id),

    # Urban
    CopernicusGlobalLandCover.Urban_and_built_up.id: (
        WteLandCover.Settlement.id),

    # Bare and sparsely vegetated areas
    CopernicusGlobalLandCover.Bare_sparse_vegetation.id: (
        WteLandCover.Bare_areas.id),

    # Snow and ice
    CopernicusGlobalLandCover.Snow_and_ice.id: WteLandCover.Snow_and_Ice.id,

    # Water bodies
    CopernicusGlobalLandCover.Water_bodies.id: WteLandCover.Water.id,

    # Wetlands
    CopernicusGlobalLandCover.Herbaceous_wetland.id: WteLandCover.NoData.id,

    # Moss and lichen
    CopernicusGlobalLandCover.Moss_and_lichen.id: WteLandCover.Bare_areas.id,

    # Forest
    CopernicusGlobalLandCover.Closed_evergreen_needleleaf_forest.id: (
        WteLandCover.Forest.id),
    CopernicusGlobalLandCover.Closed_evergreen_broadleaf_forest.id: (
        WteLandCover.Forest.id),
    CopernicusGlobalLandCover.Closed_deciduous_needleleaf_forest.id: (
        WteLandCover.Forest.id),
    CopernicusGlobalLandCover.Closed_deciduous_broadleaf_forest.id: (
        WteLandCover.Forest.id),
    CopernicusGlobalLandCover.Closed_mixed_forest.id: WteLandCover.Forest.id,
    CopernicusGlobalLandCover.Closed_unknown_forest.id: WteLandCover.Forest.id,
    CopernicusGlobalLandCover.Open_evergreen_needleleaf_forest.id: (
        WteLandCover.Forest.id),
    CopernicusGlobalLandCover.Open_evergreen_broadleaf_forest.id: (
        WteLandCover.Forest.id),
    CopernicusGlobalLandCover.Open_deciduous_needleleaf_forest.id: (
        WteLandCover.Forest.id),
    CopernicusGlobalLandCover.Open_deciduous_broadleaf_forest.id: (
        WteLandCover.Forest.id),
    CopernicusGlobalLandCover.Open_mixed_forest.id: WteLandCover.Forest.id,
    CopernicusGlobalLandCover.Open_unknown_forest.id: WteLandCover.Forest.id,

    # Open sea
    CopernicusGlobalLandCover.Open_sea.id: WteLandCover.Water.id,

    # No data
    CopernicusGlobalLandCover.NoData.id: WteLandCover.NoData.id

    })


class Globcover(Label):
    """The `Globcover`_ land cover legend.

    .. _Globcover:
        http://due.esrin.esa.int/page_globcover.php

    """

    Irrigated_cropland = 11, (170, 240, 240)
    Rainfed_cropland = 14, (255, 255, 100)
    Mosaic_cropland = 20, (220, 240, 100)
    Mosaic_vegetation = 30, (205, 205, 102)
    Broadleafed_evergreen_or_semi_deciduous_forest = 40, (0, 100, 0)
    Closed_broadleaved_deciduous_forest = 50, (0, 160, 0)
    Open_broadleaved_deciduous_forest = 60, (170, 200, 0)
    Closed_needleleaved_evergreen_forest = 70, (0, 60, 0)
    Open_needleleaved_deciduous_or_evergreen_forest = 90, (40, 100, 0)
    Mixed_broadleaved_and_needleleaf_forest = 100, (120, 130, 0)
    Mosaic_forest_and_shrubland_and_grassland = 110, (140, 160, 0)
    Mosaic_grassland_and_forest_and_shrubland = 120, (190, 150, 0)
    Closed_to_open_shrubland = 130, (150, 100, 0)
    Closed_to_open_herbaceous_vegetation = 140, (255, 180, 50)
    Sparse_vegetation = 150, (255, 235, 175)
    Broadleaved_forest_regularly_flooded = 160, (0, 120, 90)
    Broadleaved_forest_or_shrubland_permanently_flooded = 170, (0, 150, 120)
    Grassland_or_woody_vegetation_regularly_flooded = 180, (0, 220, 130)
    Artificial_surfaces_and_associated_areas = 190, (195, 20, 0)
    Bare_areas = 200, (255, 245, 215)
    Water_bodies = 210, (0, 70, 200)
    Permanent_snow_and_ice = 220, (255, 255, 255)
    NoData = 255, (0, 0, 0)  # 230: in original product


# Class label mapping from the Globcover legend, which is based on the United
# Nations Land Cover Classification System (LCCS), to WTE Land Cover
Globcover2Wte = LabelMapping({

    # LCCS A11: Cultivated Terrestrial areas and managed lands
    Globcover.Irrigated_cropland.id: WteLandCover.Cropland.id,
    Globcover.Rainfed_cropland.id: WteLandCover.Cropland.id,
    Globcover.Mosaic_cropland.id: WteLandCover.Cropland.id,
    Globcover.Mosaic_vegetation.id: WteLandCover.Cropland.id,

    # LCCS A12: Natural and semi-natural terrestrial vegetation
    Globcover.Broadleafed_evergreen_or_semi_deciduous_forest.id: (
        WteLandCover.Forest.id),
    Globcover.Closed_broadleaved_deciduous_forest.id: WteLandCover.Forest.id,
    Globcover.Open_broadleaved_deciduous_forest.id: WteLandCover.Forest.id,
    Globcover.Closed_needleleaved_evergreen_forest.id: WteLandCover.Forest.id,
    Globcover.Open_needleleaved_deciduous_or_evergreen_forest.id: (
        WteLandCover.Forest.id),
    Globcover.Mixed_broadleaved_and_needleleaf_forest.id: (
        WteLandCover.Forest.id),
    Globcover.Mosaic_forest_and_shrubland_and_grassland.id: (
        WteLandCover.Forest.id),
    Globcover.Mosaic_grassland_and_forest_and_shrubland.id: (
        WteLandCover.Shrubland.id),
    Globcover.Closed_to_open_shrubland.id: WteLandCover.Shrubland.id,
    Globcover.Closed_to_open_herbaceous_vegetation.id: (
        WteLandCover.Grassland.id),
    Globcover.Sparse_vegetation.id: WteLandCover.Bare_areas.id,

    # LCCS A24: Natural and semi-natural aquatic vegetation
    Globcover.Broadleaved_forest_regularly_flooded.id: WteLandCover.Forest.id,
    Globcover.Broadleaved_forest_or_shrubland_permanently_flooded.id: (
        WteLandCover.Forest.id),
    Globcover.Grassland_or_woody_vegetation_regularly_flooded.id: (
        WteLandCover.Shrubland.id),

    # LCCS B15: Artificial surfaces
    Globcover.Artificial_surfaces_and_associated_areas.id: (
        WteLandCover.Settlement.id),

    # LCCS B16: Bare areas
    Globcover.Bare_areas.id: WteLandCover.Bare_areas.id,

    # LCCS B28: Inland water bodies and snow and ice
    Globcover.Water_bodies.id: WteLandCover.Water.id,
    Globcover.Permanent_snow_and_ice.id: WteLandCover.Snow_and_Ice.id,

    # No data
    Globcover.NoData.id: WteLandCover.NoData.id

    })


class ModisIGBP(Label):
    """The Modis `International Geosphere-Biosphere Programme`_ legend.

    .. _International Geosphere-Biosphere Programme:
        https://lpdaac.usgs.gov/products/mcd12q1v006/

    """

    Needleleaf_evergreen_forest = 1, (38, 115, 0)
    Broadleaf_evergreen_forest = 2, (82, 204, 77)
    Needleleaf_deciduous_forest = 3, (150, 196, 20)
    Broadleaf_deciduous_forest = 4, (122, 250, 166)
    Mixed_forest = 5, (137, 205, 102)
    Closed_shrublands = 6, (215, 158, 158)
    Open_shrublands = 7, (255, 240, 196)
    Woody_savannas = 8, (233, 255, 190)
    Savannas = 9, (255, 216, 20)
    Grasslands = 10, (255, 196, 120)
    Permanent_Wetlands = 11, (0, 132, 168)
    Croplands = 12, (255, 255, 115)
    Urban_and_built_up_areas = 13, (255, 0, 0)
    Mosaic_cropland_and_natural_vegetation = 14, (168, 168, 0)
    Permanent_snow_and_ice = 15, (200, 240, 255)
    Barren = 16, (130, 130, 130)
    Water_bodies = 17, (140, 219, 255)
    Unclassified = 255, (0, 0, 0)


# Class label mapping from MODIS IGBP to WTE Land Cover
ModisIGBP2Wte = LabelMapping({

    # Forests
    ModisIGBP.Needleleaf_evergreen_forest.id: WteLandCover.Forest.id,
    ModisIGBP.Broadleaf_evergreen_forest.id: WteLandCover.Forest.id,
    ModisIGBP.Needleleaf_deciduous_forest.id: WteLandCover.Forest.id,
    ModisIGBP.Broadleaf_deciduous_forest.id: WteLandCover.Forest.id,
    ModisIGBP.Mixed_forest.id: WteLandCover.Forest.id,
    ModisIGBP.Woody_savannas.id: WteLandCover.Forest.id,
    ModisIGBP.Savannas.id: WteLandCover.Forest.id,

    # Shrublands
    ModisIGBP.Closed_shrublands.id: WteLandCover.Shrubland.id,
    ModisIGBP.Open_shrublands.id: WteLandCover.Shrubland.id,

    # Wetlands
    ModisIGBP.Permanent_Wetlands.id: WteLandCover.NoData.id,

    # Grasslands
    ModisIGBP.Grasslands.id: WteLandCover.Grassland.id,

    # Croplands
    ModisIGBP.Croplands.id: WteLandCover.Cropland.id,
    ModisIGBP.Mosaic_cropland_and_natural_vegetation.id: (
        WteLandCover.Cropland.id),

    # Settlements
    ModisIGBP.Urban_and_built_up_areas.id: WteLandCover.Settlement.id,

    # Snow and ice
    ModisIGBP.Permanent_snow_and_ice.id: WteLandCover.Snow_and_Ice.id,

    # Sparsely or non-vegetated
    ModisIGBP.Barren.id: WteLandCover.Bare_areas.id,

    # Surface water
    ModisIGBP.Water_bodies.id: WteLandCover.Water.id,

    # No data
    ModisIGBP.Unclassified.id: WteLandCover.NoData.id

    })


class ModisUMD(Label):
    """The Modis `University of Maryland`_ legend.

    .. _University of Maryland:
        https://lpdaac.usgs.gov/products/mcd12q1v006/

    """

    Water_bodies = 0, (140, 219, 255)
    Needleleaf_evergreen_forest = 1, (38, 115, 0)
    Broadleaf_evergreen_forest = 2, (82, 204, 77)
    Needleleaf_deciduous_forest = 3, (150, 196, 20)
    Broadleaf_deciduous_forest = 4, (122, 250, 166)
    Mixed_forest = 5, (137, 205, 102)
    Closed_shrublands = 6, (215, 158, 158)
    Open_shrublands = 7, (255, 240, 196)
    Woody_savannas = 8, (233, 255, 190)
    Savannas = 9, (255, 216, 20)
    Grasslands = 10, (255, 196, 120)
    Permanent_Wetlands = 11, (0, 132, 168)
    Croplands = 12, (255, 255, 115)
    Urban_and_built_up_areas = 13, (255, 0, 0)
    Mosaic_cropland_and_natural_vegetation = 14, (168, 168, 0)
    Non_vegetated_lands = 15, (130, 130, 130)
    Unclassified = 255, (0, 0, 0)


# Class label mapping from MODIS UMD to WTE Land Cover
ModisUMD2Wte = LabelMapping({

    # Forests
    ModisUMD.Needleleaf_evergreen_forest.id: WteLandCover.Forest.id,
    ModisUMD.Broadleaf_evergreen_forest.id: WteLandCover.Forest.id,
    ModisUMD.Needleleaf_deciduous_forest.id: WteLandCover.Forest.id,
    ModisUMD.Broadleaf_deciduous_forest.id: WteLandCover.Forest.id,
    ModisUMD.Mixed_forest.id: WteLandCover.Forest.id,
    ModisUMD.Woody_savannas.id: WteLandCover.Forest.id,
    ModisUMD.Savannas.id: WteLandCover.Forest.id,

    # Shrublands
    ModisUMD.Closed_shrublands.id: WteLandCover.Shrubland.id,
    ModisUMD.Open_shrublands.id: WteLandCover.Shrubland.id,

    # Wetlands
    ModisUMD.Permanent_Wetlands.id: WteLandCover.NoData.id,

    # Grasslands
    ModisUMD.Grasslands.id: WteLandCover.Grassland.id,

    # Croplands
    ModisUMD.Croplands.id: WteLandCover.Cropland.id,
    ModisUMD.Mosaic_cropland_and_natural_vegetation.id: (
        WteLandCover.Cropland.id),

    # Settlements
    ModisUMD.Urban_and_built_up_areas.id: WteLandCover.Settlement.id,

    # Sparsely or non-vegetated
    ModisUMD.Non_vegetated_lands.id: WteLandCover.Bare_areas.id,

    # Surface water
    ModisUMD.Water_bodies.id: WteLandCover.Water.id,

    # No data
    ModisUMD.Unclassified.id: WteLandCover.NoData.id

    })


class Globeland30(Label):
    """The `Globeland30`_ legend.

    .. _Globeland30:
        http://www.globallandcover.com/Page/EN_sysFrame/dataIntroduce.html?columnID=81&head=product&para=product&type=data

    """

    NoData = 0, (0, 0, 0)
    Cultivated_land = 10, (250, 160, 255)
    Forest = 20, (0, 100, 0)
    Grassland = 30, (100, 255, 0)
    Shrubland = 40, (0, 255, 120)
    Wetland = 50, (0, 100, 255)
    Water_bodies = 60, (0, 0, 255)
    Tundra = 70, (100, 100, 50)
    Artificial_surfaces = 80, (255, 0, 0)
    Bare_land = 90, (190, 190, 190)
    Permanent_snow_and_ice = 100, (200, 240, 255)
    Open_sea = 255, (0, 200, 255)


# Class label mapping from Globeland30 to WTE Land Cover
Globeland2Wte = LabelMapping({

    Globeland30.Cultivated_land.id: WteLandCover.Cropland.id,
    Globeland30.Forest.id: WteLandCover.Forest.id,
    Globeland30.Grassland.id: WteLandCover.Grassland.id,
    Globeland30.Shrubland.id: WteLandCover.Shrubland.id,
    Globeland30.Wetland.id: WteLandCover.NoData.id,
    Globeland30.Water_bodies.id: WteLandCover.Water.id,
    Globeland30.Tundra.id: WteLandCover.Bare_areas.id,
    Globeland30.Artificial_surfaces.id: WteLandCover.Settlement.id,
    Globeland30.Bare_land.id: WteLandCover.Bare_areas.id,
    Globeland30.Permanent_snow_and_ice.id: WteLandCover.Snow_and_Ice.id,
    Globeland30.Open_sea.id: WteLandCover.NoData.id,
    Globeland30.NoData.id: WteLandCover.NoData.id

    })


class LISS(Label):
    """The `Land Information System South Tyrol`_ legend.

    .. _Land Information System South Tyrol:
        http://www.eurac.edu/en/research/projects/Pages/projectdetails.aspx?pid=10200

    """

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


# Class label mapping from LISS to WTE Land Cover
LISS2Wte = LabelMapping({

    LISS.Artificial_surfaces.id: WteLandCover.Settlement.id,
    LISS.Farmland.id: WteLandCover.Cropland.id,
    LISS.Vineyards.id: WteLandCover.Cropland.id,
    LISS.Fruit_plantations.id: WteLandCover.Cropland.id,
    LISS.Intensively_used_meadows_and_pastures.id: WteLandCover.Grassland.id,
    LISS.Mixed_agricultural_areas.id: WteLandCover.Cropland.id,
    LISS.Forest.id: WteLandCover.Forest.id,
    LISS.Extensively_used_pastures_and_natural_meadows.id: (
        WteLandCover.Grassland.id),
    LISS.Krummholz.id: WteLandCover.Shrubland.id,
    LISS.Dwarf_shrubs.id: WteLandCover.Shrubland.id,
    LISS.Grassland_with_trees.id: WteLandCover.Grassland.id,
    LISS.Bare_rock.id: WteLandCover.Bare_areas.id,
    LISS.Sparse_vegetation.id: WteLandCover.Bare_areas.id,
    LISS.Glaciers.id: WteLandCover.Snow_and_Ice.id,
    LISS.Wetlands.id: WteLandCover.NoData.id,
    LISS.Water_bodies.id: WteLandCover.Water.id

    })


class Sentinel2GLC(Label):
    """The `Sentinel-2 Global Land Cover`_ legend.

    .. _Sentinel-2 Global Land Cover:
        http://s2glc.cbk.waw.pl/

    """

    Clouds = 0, (255, 255, 255, 255)
    Artifical_surfaces = 62, (210, 0, 0, 255)
    Cultivated_areas = 73, (253, 211, 39, 255)
    Vineyards = 75, (176, 91, 16, 255)
    Broadleaf_tree_cover = 82, (35, 152, 0, 255)
    Coniferous_tree_cover = 83, (8, 98, 0, 255)
    Herbaceous_vegetation = 102, (249, 150, 39, 255)
    Moors_and_heathland = 103, (141, 139, 0, 255)
    Sclerophyllous_vegetation = 104, (95, 53, 6, 255)
    Marshes = 105, (149, 107, 196, 255)
    Peatbogs = 106, (77, 37, 106, 255)
    Natural_material_surfaces = 121, (154, 154, 154, 255)
    Permanent_snow_cover = 123, (106, 255, 255, 255)
    Water_bodies = 162, (20, 69, 249, 255)
    NoData = 255, (0, 0, 0, 0)


# Class label mapping from S2GLC to WTE Land Cover
S2GLC2Wte = LabelMapping({

    Sentinel2GLC.Clouds.id: WteLandCover.NoData.id,
    Sentinel2GLC.Artifical_surfaces.id: WteLandCover.Settlement.id,
    Sentinel2GLC.Cultivated_areas.id: WteLandCover.Cropland.id,
    Sentinel2GLC.Vineyards.id: WteLandCover.Cropland.id,
    Sentinel2GLC.Broadleaf_tree_cover.id: WteLandCover.Forest.id,
    Sentinel2GLC.Coniferous_tree_cover.id: WteLandCover.Forest.id,
    Sentinel2GLC.Herbaceous_vegetation.id: WteLandCover.Grassland.id,
    Sentinel2GLC.Moors_and_heathland.id: WteLandCover.Shrubland.id,
    Sentinel2GLC.Sclerophyllous_vegetation.id: WteLandCover.Shrubland.id,
    Sentinel2GLC.Marshes.id: WteLandCover.NoData.id,
    Sentinel2GLC.Peatbogs.id: WteLandCover.NoData.id,
    Sentinel2GLC.Natural_material_surfaces.id: WteLandCover.Bare_areas.id,
    Sentinel2GLC.Permanent_snow_cover.id: WteLandCover.Snow_and_Ice.id,
    Sentinel2GLC.Water_bodies.id: WteLandCover.Water.id,
    Sentinel2GLC.NoData.id: WteLandCover.NoData.id

    })


class HumboldtLC(Label):
    """The `Humboldt University Land Cover`_ legend.

    .. _Humboldt University Land Cover:
        https://doi.pangaea.de/10.1594/PANGAEA.896282

    """

    Artifical_surfaces = 1, (255, 0, 0)
    Cropland_seasonal = 2, (255, 255, 0)
    Cropland_perennial = 3, (180, 180, 0)
    Forest_broadleaved = 4, (0, 200, 0)
    Forest_coniferous = 5, (0, 100, 100)
    Forest_mixed = 6, (0, 100, 0)
    Shrubland = 7, (255, 100, 0)
    Grassland = 8, (200, 200, 100)
    Barren = 9, (200, 200, 200)
    Water = 10, (0, 0, 255)
    Wetland = 11, (200, 0, 200)
    Snow_and_ice = 12, (255, 255, 255)
    NoData = 255, (0, 0, 0)


# Class label mapping from HULC to WTE Land Cover
Hulc2Wte = LabelMapping({

    HumboldtLC.Artifical_surfaces.id: WteLandCover.Settlement.id,
    HumboldtLC.Cropland_seasonal.id: WteLandCover.Cropland.id,
    HumboldtLC.Cropland_perennial.id: WteLandCover.Cropland.id,
    HumboldtLC.Forest_broadleaved.id: WteLandCover.Forest.id,
    HumboldtLC.Forest_coniferous.id: WteLandCover.Forest.id,
    HumboldtLC.Forest_mixed.id: WteLandCover.Forest.id,
    HumboldtLC.Shrubland.id: WteLandCover.Shrubland.id,
    HumboldtLC.Grassland.id: WteLandCover.Grassland.id,
    HumboldtLC.Barren.id: WteLandCover.Bare_areas.id,
    HumboldtLC.Water.id: WteLandCover.Water.id,
    HumboldtLC.Wetland.id: WteLandCover.NoData.id,
    HumboldtLC.Snow_and_ice.id: WteLandCover.Snow_and_Ice.id,
    HumboldtLC.NoData.id: WteLandCover.NoData.id

    })

# land cover datasets
LC_DATASET_NAMES = ['WTE', 'ESACCI', 'COPERNICUS_GLOBAL_LC', 'GLOBCOVER',
                    'GLOBELAND', 'IGBP', 'UMD', 'CORINE', 'S2GLC', 'LISS',
                    'HULC']

# list of the land cover dataset legends
LC_DATASET_LEGENDS = [WteLandCover, EsaCciLc, CopernicusGlobalLandCover,
                      Globcover, Globeland30, ModisIGBP, ModisUMD,
                      CorineLandCover, Sentinel2GLC, LISS, HumboldtLC]

# list of the land cover dataset legend conversions to WTE
LC_DATASET_LOOKUPS = [Wte2Wte, EsaCciLc2Wte, Copernicus2Wte, Globcover2Wte,
                      Globeland2Wte, ModisIGBP2Wte, ModisUMD2Wte, Corine2Wte,
                      S2GLC2Wte, LISS2Wte, Hulc2Wte]

# dictionary of the different land cover datasets
LC_LEGENDS = {k: v for k, v in zip(LC_DATASET_NAMES, LC_DATASET_LEGENDS)}

# dictionary of the different land cover dataset legend conversions
LC_LOOKUPS = {k: v for k, v in zip(LC_DATASET_NAMES, LC_DATASET_LOOKUPS)}
