#!/bin/bash

#SBATCH -N 1
#SBATCH -p All

python -u main/train_crawler_ppo.py --runs 5 --params random_exploration.json 
