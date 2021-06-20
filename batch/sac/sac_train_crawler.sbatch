#!/bin/sh
#SBATCH -p All
#SBATCH --task=1
#SBATCH --cpus-per-task=4

set NAME_VAR = %1
set /A EPISODE_VAR = %2

python3 -u ../autonome-systeme/main/train_crawler_sac.py --name %NAME_VAR% --episodes %EPISODE_VAR%