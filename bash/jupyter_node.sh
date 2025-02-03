#!/bin/bash

#$ -l tmem=50G
#$ -l gpu=true
#$ -pe gpu 1
#$ -R y
#$ -l h_rt=48:00:00
#$ -j y
#$ -N 'jupyter_notebook'

source /share/apps/source_files/python/python-3.11.9.source
source /share/apps/source_files/cuda/cuda-11.2.source
#source /share/apps/source_files/conda.source

#conda activate /SAN/colcc/WSI_LymphNodes_BreastCancer/Mengyuan/share/cell2location_env/


XDG_RUNTIME_DIR=""
host=$(hostname)
node=$(hostname -s)
user=$(whoami)
port=8879

echo "host: $host"
 
# print tunneling instructions jupyter-log
echo -e "
Command to create ssh tunnel:
ssh -N -f -L ${port}:${host}:${port} ${user}@gamble.cs.ucl.ac.uk
ssh -l ${user} -J ${user}@knuckles.cs.ucl.ac.uk -L ${port}:localhost:${port} ${node}.cs.ucl.ac.uk
 
Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"
 
 
# Run Jupyter
jupyter notebook --no-browser --port=${port} --ip=${host}
