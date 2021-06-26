#!/bin/bash

#SBATCH -N 1
#SBATCH -p All

python3 -u autonome-systeme/main/train_crawler_ppo.py --runs 1 --params test_decay_params.json 
