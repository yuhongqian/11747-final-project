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
dataset="mutual"
model_name="bert-large-uncased"
#model_name="google/electra-${size}-discriminator"

python main.py  \
  --train  \
  --eval   \
  --train_batch_size 2  \
  --grad_accumulation_steps 32  \
  --model_name ${model_name}  \
  --output_dir "${dataset}-numnet-electra-${size}-ckpts"  \
  --data_dir "../MuTual/data/${dataset}" \
  --eval_steps 500  \
  --numnet_model "/bos/usr0/hongqiay/numnet_plus/numnet_plus_345_LR_5e-4_BLR_1.5e-5_WD_5e-5_BWD_0.01tag_mspan/checkpoint_best.pt"