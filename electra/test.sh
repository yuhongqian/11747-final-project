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
epoch=1
global_step=24176
source activate cast
CUDA_VISIBLE_DEVICES=0, python main.py  \
  --test  \
  --train_batch_size 16  \
  --model_name  "google/electra-${size}-discriminator"   \
  --local_model_path "electra-${size}-ckpts-1ksteps/epoch${epoch}_global-step${global_step}"  \
  --output_dir "electra-${size}-ckpts-1ksteps/epoch${epoch}_global-step${global_step}_outputs"  \
