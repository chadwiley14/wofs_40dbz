#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=3:00:00
#SBATCH --job-name=patch_data_inter
#SBATCH --mail-user=chadwiley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/chadwiley/research/wofs_ml_ci/wofs_ml/slurmouts/R-%x_%a.out
#SBATCH --error=/home/chadwiley/research/wofs_ml_ci/wofs_ml/slurmouts/R-%x_%a.err
#SBATCH --array=9,10

source /home/chadwiley/.bashrc
bash

conda activate tf

python -u patch_splitter.py --run_num $SLURM_ARRAY_TASK_ID
