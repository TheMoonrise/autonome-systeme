#!/bin/sh
#SBATCH -p All
#SBATCH --task=1

python3 -u ../autonome-systeme/main/train_crawler_sac.py --fname testRun --params default_params.json