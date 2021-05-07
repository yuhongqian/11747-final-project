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

size="large"
epoch=3
global_step=56528
dataset="mutual"
model_name="bert-${size}-uncased"
python main.py  \
  --test  \
  --data_dir "../MuTual/data/${dataset}"   \
  --train_batch_size 16  \
  --model_name  ${model_name}  \
  --numnet_model "${dataset}-numnet-electra-${size}-ckpts/epoch${epoch}_global-step${global_step}"  \
  --output_dir "${dataset}-numnet-electra-${size}-ckpts/epoch${epoch}_global-step${global_step}_output"
