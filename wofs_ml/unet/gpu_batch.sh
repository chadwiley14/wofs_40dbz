#!/bin/bash
#SBATCH --partition=ai2es
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=gpu_install
#SBATCH --mail-user=chadwiley@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/chadwiley/research/wofs_ml_ci/wofs_ml/slurmouts/R-%x.%j.out
#SBATCH --error=/home/chadwiley/research/wofs_ml_ci/wofs_ml/slurmouts/R-%x.%j.err


#THIS SCRIPT ASSUMES YOU ALREADY HAVE AN ENV NAMED
#tf_gpu AND YOU ALREADY HAVE MAMBA

#need to source your bash script to access your python!
source /home/chadwiley/.bashrc
bash

conda activate tf

#use mamba to install tensorflow with the right GPU stuff
conda install -c conda-forge -y tensorflow==2.10.0=cuda112*