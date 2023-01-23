#!/bin/bash
#SBATCH --partition=large_mem
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --job-name=3d_examples_dataset
#SBATCH --mail-user=chadwiley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/chadwiley/research/wofs_40dbz/wofs_ml/slurmouts/R-%x_%a.out
#SBATCH --error=/home/chadwiley/research/wofs_40dbz/wofs_ml/slurmouts/R-%x_%a.err
#SBATCH --array=0-2%1


source /home/chadwiley/.bashrc
bash

conda activate tf

python -u data_3d_creator.py --run_num $SLURM_ARRAY_TASK_ID