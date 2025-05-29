#!/bin/bash

#$ -l tmem=70G
#$ -l gpu=true
#$ -pe gpu 1
#$ -R y
#$ -l h_rt=60:00:00
#$ -j y
#$ -N 'predict_sge'

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

input_path="/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/features"
truth_path="/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/truthlabels/gene_panel_cd45.csv"
output_path="/SAN/colcc/WSI_LymphNodes_BreastCancer/HollyR/data/LEAP/output"


python3 /home/hrafique/multimodal/src/predict_sge.py -ip $input_path -tp $truth_path -op $output_path -ff 'uni-CD45+_features.csv' -s 'CD45+'


date
