#!/bin/bash

# activate conda environment
conda activate pysegcnn

# change to project directory
cd ~/git/ai4ebv

# reference years
years=(2015 2016 2017 2018 2019 2020)

# iterate over the reference years
for y in ${years[@]}; do

    # change reference year in config file
    sed -i "s/^YEAR\s*=.*/YEAR=$y/" ./ai4ebv/main/config.py
    
    # run tile-based processing: sklearn
    # python ./ai4ebv/main/sklearn/lc_downscale.py

    # run batch-based processing: sklearn
    python ./ai4ebv/main/sklearn/lc_downscale_batch.py

    # use raw time series with torch classifiers
    # sed -i "s/^FEATURES\s*=.*/FEATURES=False/" ./ai4ebv/main/config.py

    # run tile-based processing: torch
    # python ./ai4ebv/main/torch/lc_downscale.py

    # run batch-based processing: torch
    # python ./ai4ebv/main/torch/lc_downscale_batch_train.py
    # python ./ai4ebv/main/torch/lc_downscale_batch_infer.py

done

# shutdown and deallocate virtual machine after processing
az vm deallocate -g AI4EBV_group -n $HOSTNAME
