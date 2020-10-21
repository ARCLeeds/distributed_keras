#!/bin/bash

# Bede GPU submission

#SBATCH --account=bdlds01
#SBATCH --time=1:0:0

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4

module load cuda
module load Anaconda3

nvidia-smi

source activate wmlce_env

python model/main.py
