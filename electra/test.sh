#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH --time=0 # No time limit
#SBATCH --mail-user=hongqiay@andrew.cmu.edu
#SBATCH --mail-type=END
#SBATCH -p gpu
#SBATCH --gres=gpu:1  # use 1 gpu

size="small"
epoch=6
global_step=11932
source activate cast
CUDA_VISIBLE_DEVICES=0, python main.py  \
  --test  \
  --train_batch_size 16  \
  --model_name  "google/electra-${size}-discriminator"   \
  --local_model_path "mutual-plus-electra-${size}-ckpts/epoch${epoch}_global-step${global_step}"  \
  --output_dir "mutual-plus-electra-${size}-ckpts/epoch${epoch}_global-step${global_step}_outputs"  \
