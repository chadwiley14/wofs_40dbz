#!/bin/bash
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=inter_patch_maker
#SBATCH --mail-user=chadwiley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/chadwiley/research/wofs_ml_ci/wofs_ml/slurmouts/R-%x_%a.out
#SBATCH --error=/home/chadwiley/research/wofs_ml_ci/wofs_ml/slurmouts/R-%x_%a.err
#SBATCH --array=1-4

source /home/chadwiley/.bashrc
bash

conda activate tf

python -u patcher_main.py --lead_time '06' --year '2020' --run_num $SLURM_ARRAY_TASK_ID