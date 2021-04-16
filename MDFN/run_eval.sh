#!/usr/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G 
#SBATCH --time=0 # No time limit
#SBATCH -p gpu
#SBATCH --gres=gpu:1  # use 1 gpu

module load anaconda3
source activate conv_dia_2

python error_analysis.py
