"""Train a scikit-learn classifier for each tile."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# change nodata id replacemnet to current class label id in line number 226 and 281. Also change import from landcover to whatever landcover is being used.
# builtins
import re
import time
import logging
from logging.config import dictConfig
from datetime import timedelta

# externals
import numpy as np
import xarray as xr
from osgeo import gdal
import os, glob
import pathlib

# locals
from ai4ebv.core.dataset import HLSDataset
from ai4ebv.core.sample import TrainingDataFactory
from ai4ebv.core.legend import Legend
from ai4ebv.core.landcover import LISS
from ai4ebv.core.predict import predict_hls_tile
from ai4ebv.core.utils import mosaic_tiles
from ai4ebv.core.constants import (ALPS_TILES, STT_TILES, HIMALAYAS_SITE,
                                   TILE_PATTERN, ALPS_CRS, HIMALAYAS_CRS)
from ai4ebv.main.io import (ROOT, ROOTmy, ASC_mask, DSC_mask, TS_PATH, MASK_PATH, CLASS_PATH, TRAIN_PATH, LC_LAYERS,
                            DEM_LAYERS)
from ai4ebv.main.config import (TILES, YEAR, MONTHS, NPIXEL, BUFFER, DROPNA,
                                DROPQA, OVERWRITE_TIME_SERIES, LC_LABELS,
                                AZURE, USE_INDICES, DEM, APPLY_QA, MOSAIC,
                                OVERWRITE_TRAINING_DATA, SEASONAL, satellite, orbit, product)
from ai4ebv.main.sklearn_config import clf, TILE_SIZE, SKLEARN_OVERWRITE
from pysegcnn.core.logging import log_conf
from pysegcnn.core.trainer import LogConfig
from pysegcnn.core.utils import np2tif, array_replace

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # initialize timing
    start_time = time.monotonic()

    # -------------------------------------------------------------------------
    # INITIALIZE LOGGING ------------------------------------------------------
    # -------------------------------------------------------------------------
    logfile = ROOTmy.joinpath('log.txt')

    # remove existing log-files
    if logfile.exists():
        logfile.unlink()

    # initialize logging
    dictConfig(log_conf(logfile))

    # -------------------------------------------------------------------------
    # INITIALIZE LOOP OVER TILES ----------------------------------------------
    # -------------------------------------------------------------------------

    # list of output files: required to generate mosaic
    y_layers = []
    p_layers = []
    q_layers = []
    for tile in TILES:
        LogConfig.init_log('Processing tile {}.'.format(tile))

        # ---------------------------------------------------------------------
        # OUTPUT FILE NAMES ---------------------------------------------------
        # ---------------------------------------------------------------------

        # classifier file name
        state = HLSDataset.state_file(
            #model=clf.estimator['clf'].__class__.__name__, labels=LC_LABELS,
            model=clf.__name__, labels=LC_LABELS,
            npixel=NPIXEL, features=False, use_indices=USE_INDICES, dem=DEM,
            qa=APPLY_QA, mode='single', months=MONTHS, year=YEAR,
            seasonal=SEASONAL)

        # check if output paths exists
        tpath = CLASS_PATH.joinpath('v1.0', tile, str(YEAR))
        dpath = TRAIN_PATH.joinpath('v1.0', tile, str(YEAR))
        for path in [tpath, dpath]:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

        # generate filenames: classified product
        fname = tpath.joinpath(state.replace('.pt', '_{}.tif'.format(tile)))
        pname = tpath.joinpath(fname.name.replace('.tif', '_prob.tif'))
        qname = dpath.joinpath(HLSDataset.time_series(tile, str(YEAR), MONTHS).
                               replace('.nc',
                                       '_'.join(['', LC_LABELS, 'qa.tif'])))

        # check if output layers exist and whether to overwrite
        if fname.exists() and not SKLEARN_OVERWRITE:
            # store predictions and associated probabilities to generate mosaic
            LOGGER.info('{} already exists. Aborting classification ...'
                        .format(fname))
            y_layers.append(fname)
            p_layers.append(pname)
            q_layers.append(qname)
            continue

        # ---------------------------------------------------------------------
        # HLS TIME SERIES -----------------------------------------------------
        # ---------------------------------------------------------------------

        # check if time series exists on disk
        if satellite == 'Sentinel-2':
            tile = "32TQS"
            time_series = TS_PATH.joinpath(satellite, tile, str(YEAR))
            time_series = str(time_series)
            print(time_series)
            s2_netcdfs = sorted(glob.glob(time_series + '/*2021.nc'))
            for i,f in enumerate(s2_netcdfs):
                #print(i,f)
                monthly_data = xr.open_dataarray(f,decode_coords="all",chunks={'time':1,'x':1000,'y':1000})
                if i == 0:
                    yearly_data_S2 = monthly_data
                    pass
                else:
                    yearly_data_S2 = xr.concat([yearly_data_S2,monthly_data],dim='time')
                    yearly_data_S2 = yearly_data_S2.sortby('time')
            dataset = yearly_data_S2.to_dataset(dim = 'variable')
            var_dict = dict({'B02': 'blue', 'B03': 'green', 'B04': 'red', 'B08': 'nir', 'B11': 'swir1', 'B12': 'swir2'})
            dataset = dataset.rename(var_dict)
            dataset = dataset.sortby('time')
            # dataset = dataset[dict(x=slice(0, 2500))]
            # dataset = dataset[dict(y=slice(0, 2500))]
            # dataset['x'] = np.arange(0,len(dataset.x))
            # dataset['y'] = np.arange(0,len(dataset.y))
            
        else:
            if product == 'COH':
                print('stacking coherence')
                
                tile = 'ST'
                coh = 'COH'
                orbit = 'DSC'
                time_series = TS_PATH.joinpath(satellite, orbit, coh, tile, str(YEAR))
                time_series = str(time_series)
                print(time_series)
                s1_netcdfs_coh = sorted(glob.glob(time_series + '/ST*.nc'))
                for i,f in enumerate(s1_netcdfs_coh):
                    monthly_data = xr.open_dataarray(f,decode_coords="all",chunks={'time':1,'x':1000,'y':1000})
                    if i == 0:
                        yearly_data_coh = monthly_data
                        pass
                    else:
                        yearly_data_coh = xr.concat([yearly_data_coh,monthly_data],dim='time')
                dataset = yearly_data_coh.sortby('time')
                dataset = dataset.to_dataset(dim = 'variable')
                MASK_PATH = DSC_mask
                mask = xr.open_rasterio(MASK_PATH)
                mask = mask[0]
                dataset_DSC = dataset*mask
                
                coh = 'COH'
                orbit = 'ASC'
                tile = 'ST'
                time_series = TS_PATH.joinpath(satellite, orbit, coh, tile, str(YEAR))
                time_series = str(time_series)
                s1_netcdfs_coh = sorted(glob.glob(time_series + '/ST*.nc'))
                for i,f in enumerate(s1_netcdfs_coh):
                    monthly_data = xr.open_dataarray(f,decode_coords="all",chunks={'time':1,'x':1000,'y':1000})
                    if i == 0:
                        yearly_data_coh = monthly_data
                        pass
                    else:
                        yearly_data_coh = xr.concat([yearly_data_coh,monthly_data],dim='time')
                dataset = yearly_data_coh.sortby('time')
                dataset = dataset.to_dataset(dim = 'variable')
                MASK_PATH = ASC_mask
                mask = xr.open_rasterio(MASK_PATH)
                mask = mask[0]
                dataset_ASC = dataset*mask
                
                fake_time = np.arange(len(dataset_ASC.time)+len(dataset_DSC.time))
                fake_time_ASC = fake_time[:len(dataset_ASC.time)]
                fake_time_DSC = fake_time[len(dataset_ASC.time):]
                dataset_ASC['time'] = fake_time_ASC
                dataset_DSC['time'] = fake_time_DSC
                dataset = xr.concat([dataset_ASC,dataset_DSC],dim='time')
                    
            elif product == 'INT':
                print('stacking intensity')
                
                tile = 'ST'
                orbit = 'DSC'
                int = 'INT'
                time_series = TS_PATH.joinpath(satellite, orbit, int, tile, str(YEAR))
                time_series = str(time_series)
                s1_netcdfs_int = sorted(glob.glob(time_series + '/ST*.nc'))
                for i,f in enumerate(s1_netcdfs_int):
                    monthly_data = xr.open_dataarray(f,decode_coords="all",chunks={'time':1,'x':1000,'y':1000})
                    if i == 0:
                        yearly_data_int = monthly_data
                        pass
                    else:
                        yearly_data_int = xr.concat([yearly_data_int,monthly_data],dim='time')
                dataset = yearly_data_int.sortby('time')
                dataset = dataset.to_dataset(dim = 'variable')
                mask = xr.open_rasterio(DSC_mask)
                mask = mask[0]
                dataset_DSC = dataset*mask
                
                tile = 'ST'
                orbit = 'ASC'
                int = 'INT'
                time_series = TS_PATH.joinpath(satellite, orbit, int, tile, str(YEAR))
                time_series = str(time_series)
                s1_netcdfs_int = sorted(glob.glob(time_series + '/ST*.nc'))
                for i,f in enumerate(s1_netcdfs_int):
                    monthly_data = xr.open_dataarray(f,decode_coords="all",chunks={'time':1,'x':1000,'y':1000})
                    if i == 0:
                        yearly_data_int = monthly_data
                        pass
                    else:
                        yearly_data_int = xr.concat([yearly_data_int,monthly_data],dim='time')
                dataset = yearly_data_int.sortby('time')
                dataset = dataset.to_dataset(dim = 'variable')
                mask = xr.open_rasterio(ASC_mask)
                mask = mask[0]
                dataset_ASC = dataset*mask
                
                fake_time = np.arange(len(dataset_ASC.time)+len(dataset_DSC.time))
                fake_time_ASC = fake_time[:len(dataset_ASC.time)]
                fake_time_DSC = fake_time[len(dataset_ASC.time):]
                dataset_ASC['time'] = fake_time_ASC
                dataset_DSC['time'] = fake_time_DSC
                dataset = xr.concat([dataset_ASC,dataset_DSC],dim='time')
                
            
            else:
                #DSC#
                tile = 'ST'
                print('stacking coherence and intensity')
                coh = 'COH'
                orbit = 'DSC'
                time_series = TS_PATH.joinpath(satellite, orbit, coh, tile, str(YEAR))
                time_series = str(time_series)
                s1_netcdfs_coh = sorted(glob.glob(time_series + '/ST*.nc'))
                for i,f in enumerate(s1_netcdfs_coh):
                    monthly_data = xr.open_dataarray(f,decode_coords="all",chunks={'time':1,'x':1000,'y':1000})
                    if i == 0:
                        yearly_data_coh = monthly_data
                        pass
                    else:
                        yearly_data_coh = xr.concat([yearly_data_coh,monthly_data],dim='time')
                yearly_data_coh = yearly_data_coh.sortby('time')    
                
                int = 'INT'
                time_series = TS_PATH.joinpath(satellite, orbit, int, tile, str(YEAR))
                time_series = str(time_series)
                s1_netcdfs_int = sorted(glob.glob(time_series + '/ST*.nc'))
                for i,f in enumerate(s1_netcdfs_int):
                    monthly_data = xr.open_dataarray(f,decode_coords="all",chunks={'time':1,'x':1000,'y':1000})
                    if i == 0:
                        yearly_data_int = monthly_data
                        pass
                    else:
                        yearly_data_int = xr.concat([yearly_data_int,monthly_data],dim='time')
                yearly_data_int = yearly_data_int.sortby('time')
                time_series = TS_PATH.joinpath(satellite, orbit, product, tile, str(YEAR))    
                fake_time = np.arange(len(yearly_data_coh.time)+len(yearly_data_int.time))
                fake_time_coh = fake_time[:len(yearly_data_coh.time)]
                fake_time_int = fake_time[len(yearly_data_coh.time):]
                yearly_data_coh['time'] = fake_time_coh
                yearly_data_int['time'] = fake_time_int
                yearly_data_coh = yearly_data_coh.to_dataset(dim='variable')
                yearly_data_coh = yearly_data_coh.rename({'coh_vh_amp':'VH','coh_vv_amp':'VV'}).to_array()
                dataset = xr.concat([yearly_data_coh,yearly_data_int],dim='time')
                dataset = dataset.to_dataset(dim = 'variable')
                MASK_PATH = DSC_mask
                mask = xr.open_rasterio(MASK_PATH)
                mask = mask[0]
                dataset_DSC = dataset*mask
                
                #ASC#
                print('stacking coherence and intensity')
                coh = 'COH'
                orbit = 'ASC'
                tile = 'ST'
                time_series = TS_PATH.joinpath(satellite, orbit, coh, tile, str(YEAR))
                time_series = str(time_series)
                s1_netcdfs_coh = sorted(glob.glob(time_series + '/ST*.nc'))
                for i,f in enumerate(s1_netcdfs_coh):
                    monthly_data = xr.open_dataarray(f,decode_coords="all",chunks={'time':1,'x':1000,'y':1000})
                    if i == 0:
                        yearly_data_coh = monthly_data
                        pass
                    else:
                        yearly_data_coh = xr.concat([yearly_data_coh,monthly_data],dim='time')
                yearly_data_coh = yearly_data_coh.sortby('time')    
                
                int = 'INT'
                time_series = TS_PATH.joinpath(satellite, orbit, int, tile, str(YEAR))
                time_series = str(time_series)
                s1_netcdfs_int = sorted(glob.glob(time_series + '/ST*.nc'))
                for i,f in enumerate(s1_netcdfs_int):
                    monthly_data = xr.open_dataarray(f,decode_coords="all",chunks={'time':1,'x':1000,'y':1000})
                    if i == 0:
                        yearly_data_int = monthly_data
                        pass
                    else:
                        yearly_data_int = xr.concat([yearly_data_int,monthly_data],dim='time')
                yearly_data_int = yearly_data_int.sortby('time')
                time_series = TS_PATH.joinpath(satellite, orbit, product, tile, str(YEAR))    
                fake_time = np.arange(len(yearly_data_coh.time)+len(yearly_data_int.time))
                fake_time_coh = fake_time[:len(yearly_data_coh.time)]
                fake_time_int = fake_time[len(yearly_data_coh.time):]
                yearly_data_coh['time'] = fake_time_coh
                yearly_data_int['time'] = fake_time_int
                yearly_data_coh = yearly_data_coh.to_dataset(dim='variable')
                yearly_data_coh = yearly_data_coh.rename({'coh_vh_amp':'VH','coh_vv_amp':'VV'}).to_array()
                dataset = xr.concat([yearly_data_coh,yearly_data_int],dim='time')
                dataset = dataset.to_dataset(dim = 'variable')
                MASK_PATH = ASC_mask
                mask = xr.open_rasterio(MASK_PATH)
                mask = mask[0]
                dataset_ASC = dataset*mask
                # dataset_ASC = dataset_ASC.drop("variable")
                # dataset_DSC = dataset_DSC.drop("variable")
                fake_time = np.arange(len(dataset_ASC.time)+len(dataset_DSC.time))
                fake_time_ASC = fake_time[:len(dataset_ASC.time)]
                fake_time_DSC = fake_time[len(dataset_ASC.time):]
                dataset_ASC['time'] = fake_time_ASC
                dataset_DSC['time'] = fake_time_DSC
                dataset = xr.concat([dataset_ASC,dataset_DSC],dim='time')
                
                tile = "32TPS"
                satellite = "Sentinel-2"
                time_series = TS_PATH.joinpath(satellite, tile, str(YEAR))
                time_series = str(time_series)
                s2_netcdfs = sorted(glob.glob(time_series + '/*2021.nc'))
                for i,f in enumerate(s2_netcdfs):
                    #print(i,f)
                    monthly_data = xr.open_dataarray(f,decode_coords="all",chunks={'time':1,'x':1000,'y':1000})
                    if i == 0:
                        yearly_data_S2 = monthly_data
                        pass
                    else:
                        yearly_data_S2 = xr.concat([yearly_data_S2,monthly_data],dim='time')
                        yearly_data_S2 = yearly_data_S2.sortby('time')
                dataset_S2 = yearly_data_S2.to_dataset(dim = 'variable')
                var_dict = dict({'B02': 'blue', 'B03': 'green', 'B04': 'red', 'B08': 'nir', 'B11': 'swir1', 'B12': 'swir2'})
                dataset_S2 = dataset_S2.rename(var_dict)
                dataset_S2 = dataset_S2.sortby('time')
                
                fake_time = np.arange(len(dataset.time)+len(dataset_S2.time))
                fake_time_S1 = fake_time[:len(dataset.time)]
                fake_time_S2 = fake_time[len(dataset.time):]
                dataset['time'] = fake_time_S1
                dataset_S2['time'] = fake_time_S2
                
                min_y = min(dataset.y)
                max_y = max(dataset_S2.y)
                min_x = min(dataset.x)
                max_x = max(dataset_S2.x)
                cropped_dataset_S2 = dataset_S2.sel(y=slice(max_y,min_y), x=slice(min_x,max_x))
                cropped_dataset_S1 = dataset.sel(y=slice(min_y,max_y), x=slice(min_x,max_x))
                dataset = xr.merge([cropped_dataset_S1,cropped_dataset_S2])
            
        
        hls_ts = dataset
        time_series = pathlib.Path(time_series)
        print(time_series)
        print(hls_ts)
        
        
        
            
        if hls_ts is None:
            continue

        # ---------------------------------------------------------------------
        # LAND COVER LABELS ---------------------------------------------------
        # ---------------------------------------------------------------------

        # get the land cover labels for the current tile
        tile = "32TPS"
        lc = LC_LAYERS[LC_LABELS][tile]
        #print(lc)
        ds = gdal.Open(str(lc))

        # ---------------------------------------------------------------------
        # DIGITAL ELEVATION MODEL ---------------------------------------------
        # ---------------------------------------------------------------------

        # get the digital elevation model
        dem = DEM_LAYERS[tile] if DEM else None
        if dem is not None:
            # dem features: elevation, slope and aspect
            dem_features = TrainingDataFactory.dem_features(dem)

        # ---------------------------------------------------------------------
        # SAMPLE TRAINING DATA ------------------------------------------------
        # ---------------------------------------------------------------------

        # generate filenames: training data
        #sname = dpath.joinpath(time_series.name.replace(
        #    time_series.suffix, '_'.join(['', LC_LABELS, 'samples.tif'])))
        sname = dpath.joinpath('LISS_samples.tif')
        tname = dpath.joinpath('LISS_train.tif')
        #tname = dpath.joinpath(time_series.name.replace(
        #    time_series.suffix, '_'.join(['', LC_LABELS, 'train.tif'])))
        dname = dpath.joinpath(time_series.name.replace(
            time_series.suffix, '_'.join(['', LC_LABELS, 'train.nc'])))
        qname = dpath.joinpath(time_series.name.replace(
            time_series.suffix, '_'.join(['', LC_LABELS, 'qa.tif'])))

        # check if training data exists
        if dname.exists() and not OVERWRITE_TRAINING_DATA:
            LogConfig.init_log('Existing training data: {}'.format(dname))
            training_data = xr.open_dataset(dname)
            #print(dname)
            
        else:
            # instanciate training data factory
            factory = TrainingDataFactory(hls_ts, lc, LC_LABELS, qa=APPLY_QA)
            

            # generate training and validation data
            training_data, samples = factory.generate_training_data(
                factory.hls_ts, factory.samples, factory.class_labels, NPIXEL,
                buffer=BUFFER,  tname=tname)

            # save training data spectra: NetCDF file
            LogConfig.init_log('Saving training data.')
            LOGGER.info('Saving: {}'.format(dname))
            #training_data.to_netcdf(dname, engine='h5netcdf')

            # save sampling and training data layer
            np2tif(factory.samples, filename=sname,
                   no_data=Legend.NoData.id, src_ds=ds, overwrite=True)
            np2tif(samples, filename=tname,
                   no_data=Legend.NoData.id, src_ds=ds, overwrite=True)

            # save pixel quality assessment layer: proportion of valid time
            # steps
            # np2tif(factory.qa, filename=qname, no_data=np.nan, src_ds=ds,
            #       overwrite=True)

        # ---------------------------------------------------------------------
        # TRAIN CLASSIFIER ----------------------------------------------------
        # ---------------------------------------------------------------------

        # load training data to memory
        #print(training_data)
        inputs, labels = TrainingDataFactory.load_training_data(
            training_data, features=False, use_indices=USE_INDICES)

        # add digital elevation model features
        if dem is not None:
            dem_features = dem_features.sel(
                y=training_data.y, x=training_data.x).to_array().values
            inputs = np.concatenate((inputs, dem_features.swapaxes(0, 1)),
                                    axis=1)

        # fit classifier to training data
        LogConfig.init_log('Fitting {}.'.format(clf.__class__.__name__))
        
        #Adding stupid stuff
        print(inputs.shape)
        clf = clf(oob_score=True, n_jobs=-1)
        clf.fit(np.asarray(inputs), np.asarray(labels))
        
        #clf.fit(np.asarray(inputs), np.asarray(labels))
        print(clf)

        # ---------------------------------------------------------------------
        # PREDICT TILE --------------------------------------------------------
        # ---------------------------------------------------------------------
        LogConfig.init_log('Predicting tile {}.'.format(tile))
        y_pred, y_prob = predict_hls_tile(
            hls_ts, clf, tile_size=TILE_SIZE, features=False,
            use_indices=USE_INDICES, dem=dem)
        
        
        # map working legend to WTE-land cover legend
        # y_pred_wte = array_replace(y_pred, Legend2Wte.to_numpy())
        #np2tif(y_pred, no_data=LISS.NoData.id, overwrite=True,
        #       filename=str(fname).replace(fname.suffix, '_wte.tif'),
        #       src_ds=ds)

        # ---------------------------------------------------------------------
        # SAVE PREDICTIONS AS GEOTIFF -----------------------------------------
        # ---------------------------------------------------------------------

        # save tile-wise predictions and associated probabilities as GeoTiff
        #np2tif(y_pred, filename=fname, no_data=Legend.NoData.id,
        #       src_ds=ds, overwrite=True)
        #np2tif(y_prob, filename=pname, no_data=np.nan,
        #       src_ds=ds, overwrite=True)
        y_pred.rio.to_raster(fname)
        y_prob.rio.to_raster(pname)

        # store predictions and associated probabilities to generate mosaic
        y_layers.append(fname)
        p_layers.append(pname)
        q_layers.append(qname)

    # -------------------------------------------------------------------------
    # GENERATE MOSAICS --------------------------------------------------------
    # -------------------------------------------------------------------------
    if MOSAIC:
        LogConfig.init_log('Initializing mosaicking.')

        # check spatial coverage of mosaic
        if TILES == ALPS_TILES:
            coverage = 'Alps'
        elif TILES == STT_TILES:
            coverage = 'Province'
        elif TILES == HIMALAYAS_SITE:
            coverage = 'Himalayas'
        else:
            coverage = ('Local_Alps' if set(TILES).intersection(ALPS_TILES)
                        else 'Local_Himalaya')

        # target path to save mosaics
        target = CLASS_PATH.joinpath('v1.0', 'Mosaic', coverage, str(YEAR))
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)

        # mosaic filenames
        basename = re.search('.+?(?={})'.format(TILE_PATTERN),
                             str(y_layers[0].name))[0]
        targets = [target.joinpath(basename + '.tif'),
                   target.joinpath(basename + '_prob.tif')]

        # mosaic of classifications and posterior probabilities
        crs = (ALPS_CRS if set(TILES).intersection(ALPS_TILES) else
               HIMALAYAS_CRS)
        mosaic_tiles(y_layers, p_layers, targets=targets, trg_crs=crs)

        # mosaic of classifications in Wte-legend
        mosaic_path = target.joinpath(basename + '.tif')
        #wte = TrainingDataFactory.read_labels(targets[0], 'LEGEND')
        #np2tif(mosaic_path, filename=mosaic_path, no_data=LISS.NoData.id,
        #       src_ds=gdal.Open(str(targets[0])), overwrite=True)

        # mosaic of pixel quality assessment layer
        # mosaic_tiles(q_layers, targets=target.joinpath(basename + '_qa.tif'),
        #             method='min', trg_crs=crs)

    # log execution time of script
    LogConfig.init_log('Execution time of script {}: {}'
                       .format(__file__, timedelta(seconds=time.monotonic() -
                                                   start_time)))
