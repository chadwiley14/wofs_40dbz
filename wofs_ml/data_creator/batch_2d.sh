#!/bin/bash
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --job-name=2d_examples_dataset
#SBATCH --mail-user=chadwiley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/chadwiley/research/wofs_40dbz/wofs_ml/slurmouts/R-%x_%a.out
#SBATCH --error=/home/chadwiley/research/wofs_40dbz/wofs_ml/slurmouts/R-%x_%a.err
#SBATCH --array=3


source /home/chadwiley/.bashrc
bash

conda activate tf

python -u data_2d_creator.py --run_num $SLURM_ARRAY_TASK_ID