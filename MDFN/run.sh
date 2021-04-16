#!/usr/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G 
#SBATCH --time=0 # No time limit
#SBATCH -p gpu
#SBATCH --gres=gpu:1  # use 1 gpu

module load anaconda3
source activate conv_dia_2

python run_MDFN.py \
--data_dir datasets/mutual \
--model_name_or_path \
google/electra-large-discriminator \
--model_type electra \
--task_name mutual \
--output_dir output_mutual_electra \
--cache_dir cached_models \
--max_seq_length 256 \
--do_train --do_eval \
--train_batch_size 1 \
--eval_batch_size 1 \
--learning_rate 4e-6 \
--num_train_epochs 3 \
--gradient_accumulation_steps 1 \
--local_rank -1 \
