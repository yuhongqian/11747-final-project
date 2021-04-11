#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH --time=0 # No time limit
#SBATCH --mail-user=hongqiay@andrew.cmu.edu
#SBATCH --mail-type=END
#SBATCH -p gpu
#SBATCH --gres=gpu:1  # use 1 gpu

source activate cast
CUDA_VISIBLE_DEVICES=0, python main.py  \
  --train  \
  --train_batch_size 16  \
  --grad_accumulation_steps 4