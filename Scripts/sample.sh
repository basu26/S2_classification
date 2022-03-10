#!/bin/bash

# activate conda environment
conda activate pysegcnn

# change to project directory
cd ~/git/ai4ebv

# reference years
years=(2015 2016 2017 2018 2019 2020)

# number of pixels per class and tile
npixel=10000

# window size
buffer_size=(2 3 4 5)

# auxiliary land cover product: (global, regional, local)
product=('GLOBELAND' 'CORINE' 'LISS')

# sampling scheme: (median, conservative)
scheme=('median' 'conservative')

# iterate over the reference years
for y in ${years[@]}; do

    # change reference year in config file
    sed -i "s/^YEAR\s*=.*/YEAR=$y/" ./ai4ebv/main/config.py

    # iterate over the land cover products
    for p in ${product[@]}; do

        # change land cover product in config file
        sed -i "s/^AUX_LC_LABELS\s*=.*/AUX_LC_LABELS='$p'/" ./ai4ebv/main/config.py

        # iterate over the size of the moving window
        for b in ${buffer_size[@]}; do
            # iterate over the sampling scheme
            for s in ${scheme[@]}; do
                python ai4ebv/main/utils/sample.py $npixel $b -n $s
            done
        done
    done
done

# shutdown and deallocate virtual machine after processing
az vm deallocate -g AI4EBV_group -n $HOSTNAME