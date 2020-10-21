#!/bin/sh

# sge distributed keras submission script

#$ -V -cwd

#$ -l h_rt=1:0:0
#$ -l coproc_v100=4

module load anaconda
module load cuda

nvidia-smi

source activate tf

python model/main.py
