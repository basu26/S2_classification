#!/bin/bash

# activate conda environment
conda activate pysegcnn

# change to project directory
cd ~/git/ai4ebv

# reference years
years=(2015 2016 2017 2018 2019 2020)

# area of interest
shp='/mnt/CEPH_PROJECTS/AI4EBV/BOUNDARIES/STT.shp'

# source path
src='/mnt/CEPH_PROJECTS/AI4EBV/OUTPUTS/Classified/v0.6/Mosaic/Province'

# target path
trg='/mnt/CEPH_PROJECTS/AI4EBV/DELIVERABLES/Landcover/v0.6/Province'

# clip layers to extent of Trentino-South-Tyrol
for y in ${years[@]}; do
    python ai4ebv/main/utils/clip.py $src/$y $shp -t $trg/$y -p .tif$ -o 
done
