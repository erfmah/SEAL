#!/bin/bash

#SBATCH --job-name=MyJob
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400GB
#SBATCH --time=7-00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err
#SBATCH --partition=long
#SBATCH --mail-type=all
#SBATCH --qos=overcap
#SBATCH --nodelist=cs-venus-09

# Your experiment setup logic here

#source [MinicondaPATH]/miniconda/etc/profile.d/conda.sh
conda activate mehdi

echo "Environment activated"


python baseline_inductive.py --dataset="LLGF_photos_new_semi_ind"