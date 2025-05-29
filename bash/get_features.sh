#!/bin/bash

#$ -l tmem=70G
#$ -l gpu=true
#$ -pe gpu 1
#$ -R y
#$ -l h_rt=60:00:00
#$ -j y
#$ -N 'get_features'

hostname
# Capture the hostname in a variable
host=$(hostname)
echo "host: $host"

date

#export LD_LIBRARY_PATH=/share/apps/onetbb-2021.1.1/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/share/apps/openslide-3.4.1/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export PATH=/home/hrafique/.local/bin:$PATH
source /share/apps/source_files/python/python-3.11.9.source



python3 /home/hrafique/multimodal/src/get_features_for_patches.py


date
