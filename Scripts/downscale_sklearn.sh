#!/bin/bash

# activate conda environment
conda activate pysegcnn

# change to project directory
cd ~/git/ai4ebv

# run tile-based processing: sklearn
python ./ai4ebv/main/sklearn/lc_downscale.py

# run batch-based processing: sklearn
# python ./ai4ebv/main/sklearn/lc_downscale_batch.py

# shutdown and deallocate virtual machine after processing
az vm deallocate -g AI4EBV_group -n $HOSTNAME
