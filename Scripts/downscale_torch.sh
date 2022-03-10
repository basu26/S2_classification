#!/bin/bash

# activate conda environment
conda activate pysegcnn

# change to project directory
cd ~/git/ai4ebv

# run tile-based processing: torch
python ./ai4ebv/main/torch/lc_downscale.py

# run batch-based processing: torch
python ./ai4ebv/main/torch/lc_downscale_batch_train.py
python ./ai4ebv/main/torch/lc_downscale_batch_infer.py

# shutdown and deallocate virtual machine after processing
az vm deallocate -g AI4EBV_group -n $HOSTNAME
