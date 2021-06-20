#!/bin/sh
#SBATCH -p All
#SBATCH --task=1
#SBATCH --cpus-per-task=4

python3 -u ../autonome-systeme/main/train_crawler_sac.py --name %1 --episodes %2