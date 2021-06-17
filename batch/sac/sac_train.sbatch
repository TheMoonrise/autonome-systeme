#!/bin/sh
#SBATCH -p All
#SBATCH --task=1
#SBATCH --cpus-per-task=4
python3 -u example_sac_train_pendulum.py
