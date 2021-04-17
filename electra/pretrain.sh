#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH --time=0 # No time limit
#SBATCH --mail-user=hongqiay@andrew.cmu.edu
#SBATCH --mail-type=END
#SBATCH -p gpu
#SBATCH --gres=gpu:1  # use 1 gpu
#SBATCH --nodelist=boston-2-31
size="large"
source activate cast
python main.py  \
  --pretrain  \
  --epochs 5   \
  --train_batch_size 2   \
  --grad_accumulation_steps 32  \
  --model_name "google/electra-${size}-discriminator"  \
  --output_dir "dapo-electra-${size}-ckpts"  \
  --data_dir "../pretrain_data"  \
  --eval_steps 500