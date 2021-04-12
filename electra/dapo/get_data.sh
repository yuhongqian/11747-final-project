#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH --time=0 # No time limit
#SBATCH --mail-user=hongqiay@andrew.cmu.edu
#SBATCH --mail-type=END

source activate cast
python -u get_pretrain_data.py