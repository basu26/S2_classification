#!/bin/bash

# activate conda environment
conda activate pysegcnn

# change to project directory
cd ~/git/ai4ebv

# reference years
years=(2015 2016 2017 2018 2019 2020)

# source path
src='/mnt/CEPH_PROJECTS/AI4EBV/OUTPUTS/Classified/v0.6/Mosaic/Province'

# target path
trg='/mnt/CEPH_PROJECTS/AI4EBV/DELIVERABLES/Landcover/v0.6/Province'

# clip layers to extent of Trentino-South-Tyrol
for y in ${years[@]}; do
    # check if output directory exists
    if ! [ -d "$trg/$y" ]; then
        mkdir -p $trg/$y
    fi
 
    # copy layers
    cp $src/$y/* $trg/$y/
done
