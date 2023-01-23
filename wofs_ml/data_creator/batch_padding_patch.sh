#!/bin/bash
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --job-name=patch_splitting_no_overlapping
#SBATCH --mail-user=chadwiley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/chadwiley/research/wofs_ml_ci/wofs_ml/slurmouts/R-%x_%a.out
#SBATCH --error=/home/chadwiley/research/wofs_ml_ci/wofs_ml/slurmouts/R-%x_%a.err
#SBATCH --array=5


source /home/chadwiley/.bashrc
bash

conda activate tf

python -u patch_padding.py --run_num $SLURM_ARRAY_TASK_ID