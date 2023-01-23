#!/bin/bash
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=board
#SBATCH --mail-user=chadwiley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/chadwiley/research/wofs_ml_ci/wofs_ml/slurmouts/R-%x.%j.out
#SBATCH --error=/home/chadwiley/research/wofs_ml_ci/wofs_ml/slurmouts/R-%x.%j.err

#source you python env
source /home/chadwiley/.bashrc
bash

conda activate tf

tensorboard --logdir="/ourdisk/hpc/ai2es/chadwiley/boardlogs/log1/" --port=6098 --bind_all