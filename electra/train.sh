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
source activate cast
python main.py  \
  --train  \
  --eval   \
  --train_batch_size 2  \
  --grad_accumulation_steps 32  \
  --model_name "google/electra-${size}-discriminator"  \
  --output_dir "electra-${size}-ckpts-1ksteps"  \
  --eval_steps 1000