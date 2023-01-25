#!/bin/bash
#SBATCH --partition=large_mem
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --job-name=patch_splitting_no_overlapping
#SBATCH --mail-user=chadwiley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/chadwiley/research/wofs_40dbz/wofs_ml/slurmouts/R-%x_%a.out
#SBATCH --error=/home/chadwiley/research/wofs_40dbz/wofs_ml/slurmouts/R-%x_%a.err
#SBATCH --array=11


source /home/chadwiley/.bashrc
bash

conda activate tf

python -u patch_padding.py --run_num $SLURM_ARRAY_TASK_ID