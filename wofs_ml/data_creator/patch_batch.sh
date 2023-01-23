#!/bin/bash
#SBATCH --partition=ai2es
#SBATCH --time=08:00:00
#SBATCH --job-name=patch_data_inter
#SBATCH --mail-user=chadwiley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/chadwiley/research/wofs_ml_ci/wofs_ml/slurmouts/R-%x_%a.out
#SBATCH --error=/home/chadwiley/research/wofs_ml_ci/wofs_ml/slurmouts/R-%x_%a.err
#SBATCH --array=0-19

source /home/chadwiley/.bashrc
bash

conda activate tf

python -u patching.py --run_num $SLURM_ARRAY_TASK_ID