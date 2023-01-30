#!/bin/bash
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH -w c732
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --time=49:00:00
#SBATCH --job-name=3d_hparam_inital
#SBATCH --mail-user=chadwiley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/chadwiley/research/wofs_40dbz/wofs_ml/slurmouts/R-%x.%j.out
#SBATCH --error=/home/chadwiley/research/wofs_40dbz/wofs_ml/slurmouts/R-%x.%j.err
#SBATCH --exclusive

source /home/chadwiley/.bashrc
bash

conda activate tf

python -u hparam_wofs_3d.py --logdir="/ourdisk/hpc/ai2es/chadwiley/boardlogs/log_3d/"