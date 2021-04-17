#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH --time=0 # No time limit
#SBATCH --mail-user=hongqiay@andrew.cmu.edu
#SBATCH --mail-type=END
#SBATCH -p gpu
#SBATCH --gres=gpu:1  # use 1 gpu

size="large"
epoch=2
global_step=37352
dataset="mutual"
python main.py  \
  --test  \
  --data_dir "../MuTual/data/${dataset}"   \
  --train_batch_size 16  \
  --model_name  "google/electra-${size}-discriminator"   \
  --local_model_path "mutual-plus-electra-${size}-ckpts/epoch${epoch}_global-step${global_step}"  \
  --output_dir "mutual-plus-electra-${size}-ckpts/epoch${epoch}_global-step${global_step}_output"  \
