#!/bin/bash

#$ -l tmem=70G
#$ -l gpu=true
#$ -pe gpu 1
#$ -R y
#$ -l h_rt=23:00:00
#$ -j y
#$ -N 'reg10x-testds'


#  -l h="*thig*|*doglion*|*thog*"

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


input_path="/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP"
output_path="/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/AlignedHnE"
config_path="/home/hrafique/multimodal/config/reg_params.yaml"


python3 /home/hrafique/multimodal/src/register_images.py -ip $input_path -op $output_path -m -mag 10x -cp $config_path

date
