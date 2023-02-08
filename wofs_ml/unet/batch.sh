#!/bin/bash
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH -w c733
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --job-name=hparam_inital_wbc
#SBATCH --mail-user=chadwiley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/chadwiley/research/wofs_40dbz/wofs_ml/slurmouts/R-%x.%j.out
#SBATCH --error=/home/chadwiley/research/wofs_40dbz/wofs_ml/slurmouts/R-%x.%j.err

source /home/chadwiley/.bashrc
bash

conda activate tf

python -u hparam_wofs_ml.py --logdir="/ourdisk/hpc/ai2es/chadwiley/boardlogs/log_2d_wbc/"