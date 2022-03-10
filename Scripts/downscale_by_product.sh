#!/bin/bash

# activate conda environment
conda activate pysegcnn

# change to project directory
cd ~/git/ai4ebv

# reference years
# years=(2015 2016 2017 2018 2019 2020)
# years=(2018 2019)
years=(2018)

# auxiliary land cover product: (global, regional, local)
# product=('GLOBELAND' 'CORINE' 'LISS')
product=('GLOBELAND' 'CORINE')
# product=('CORINE')

# iterate over the reference years
for y in ${years[@]}; do

    # iterate over the land cover products
    for p in ${product[@]}; do
        # change reference year in config file
        sed -i "s/^YEAR\s*=.*/YEAR=$y/" ./ai4ebv/main/config.py

        # change land cover product in config file
        sed -i "s/^AUX_LC_LABELS\s*=.*/AUX_LC_LABELS='$p'/" ./ai4ebv/main/config.py

        # use classification features with sklearn classifiers
        sed -i "s/^FEATURES\s*=.*/FEATURES=True/" ./ai4ebv/main/config.py

        # decrease number of samples per class for traditional ML methods
        sed -i "s/^NPIXEL\s*=.*/NPIXEL=5000/" ./ai4ebv/main/config.py

        # run tile-based processing: sklearn
        python ./ai4ebv/main/sklearn/lc_downscale.py

        # run batch-based processing: sklearn
        python ./ai4ebv/main/sklearn/lc_downscale_batch.py

        # use raw time series with torch classifiers
        sed -i "s/^FEATURES\s*=.*/FEATURES=False/" ./ai4ebv/main/config.py

        # increase number of samples per class for DL methods
        sed -i "s/^NPIXEL\s*=.*/NPIXEL=10000/" ./ai4ebv/main/config.py

        # run tile-based processing: torch
        python ./ai4ebv/main/torch/lc_downscale.py

        # run batch-based processing: torch
        python ./ai4ebv/main/torch/lc_downscale_batch_train.py
        python ./ai4ebv/main/torch/lc_downscale_batch_infer.py

    done
done

# shutdown and deallocate virtual machine after processing
az vm deallocate -g AI4EBV_group -n $HOSTNAME
