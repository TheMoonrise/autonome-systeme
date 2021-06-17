#!/bin/bash

#SBATCH -n 4
#SBATCH -p All
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -v

python3 -u main/train_crawler_ppo.py