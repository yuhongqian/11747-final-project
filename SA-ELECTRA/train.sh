#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH --time=0 # No time limit
#SBATCH -p gpu
#SBATCH --gres=gpu:1  # use 1 gpu

size="small"
dataset="mutual"
model_name="google/electra-${size}-discriminator"

python run_model_train.py  \
  --train  \
  --eval   \
  --train_batch_size 4  \
  --grad_accumulation_steps 64  \
  --model_name "google/electra-small-discriminator"  \
  --output_dir "mutual-numnet-small-electra-small-ckpts"  \
  --data_dir "mutual/" \
  --eval_steps 500  \
  --speaker_embeddings