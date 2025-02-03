#!/bin/bash

#$ -l tmem=50G
#$ -l gpu=true
#$ -pe gpu 2
#$ -R y
#$ -l h_rt=20:00:00
#$ -j y
#$ -N 'reg20x-fast'

hostname
# Capture the hostname in a variable
host=$(hostname)
echo "host: $host"

date

#source /share/apps/source_files/cuda/cuda-11.2.source
#export LD_LIBRARY_PATH=/share/apps/TensorRT-6.0.1.8/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/share/apps/onetbb-2021.1.1/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/share/apps/openslide-3.4.1/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export PATH=/home/hrafique/.local/bin:$PATH
source /share/apps/source_files/python/python-3.11.9.source


python3 /home/hrafique/multimodal/src/register_images.py

date
