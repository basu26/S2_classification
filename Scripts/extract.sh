#!/bin/bash

# activate conda environment
conda activate azure

# change to project directory
cd ~/git/ai4ebv

# reference years
years=(2015 2016 2017 2018 2019 2020)

# extract tiles of interest
for y in ${years[@]}; do
    # Missing tiles of the European Alps
    # python ai4ebv/main/utils/extract.py /mnt/drive/L30.$y.tar /mnt/drive/EO/HLS -t 33TWM 33TXN -y $y
    # python ai4ebv/main/utils/extract.py /mnt/drive/S30.$y.tar /mnt/drive/EO/HLS -t 33TWM 33TXN -y $y

    # Tiles of the Himalayan study site
    python ai4ebv/main/utils/extract.py /mnt/drive/L30.$y.tar /mnt/drive/EO/HLS -t 45RVM 45RVL 45RUM 45RUL -y $y
    python ai4ebv/main/utils/extract.py /mnt/drive/S30.$y.tar /mnt/drive/EO/HLS -t 45RVM 45RVL 45RUM 45RUL -y $y
done

# synchronize files to virtual machine for classification
# rsync -avhrP /mnt/drive/EO/HLS/ euracuser@51.143.8.75:/mnt/drive/AI4EBV/EO/HLS

# shutdown and deallocate virtual machine
az vm deallocate -g AI4EBV_group -n $HOSTNAME
