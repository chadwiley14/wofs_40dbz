#!/bin/bash
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=compress_data
#SBATCH --mail-user=chadwiley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/chadwiley/research/wofs_ml_ci/wofs_ml/data_compress/output/data_run_%04a_stdout.txt
#SBATCH --error=/home/chadwiley/research/wofs_ml_ci/wofs_ml/data_compress/output/data_run_%04a_stderr.txt
#SBATCH --array=24-37

source /home/chadwiley/.bashrc
bash

conda activate tf

python -u data_creator.py --file_num $SLURM_ARRAY_TASK_ID --data_path '/scratch/chadwiley/patches/examples' --save_path '/ourdisk/hpc/ai2es/chadwiley/patches/examples_norm'