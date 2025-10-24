#!/bin/bash

#SBATCH --job-name=LfH_main_04cl_20obs_adamW_adjusted_lambdas_continued
#SBATCH --qos=gpu
#SBATCH --partition=contrib-gpuq
#SBATCH --gres=gpu:1g.10gb:1

#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40GB 
#SBATCH --time=5-00:00:00

#SBATCH --output=logs/LfH_main_04cl_20obs_adamW_adjusted_lambdas_continued.%j.out
#SBATCH --error=logs/LfH_main_04cl_20obs_adamW_adjusted_lambdas_continued.%j.err

#SBATCH --mail-user=sghani2@gmu.edu
#SBATCH --mail-type=ALL

source /projects/RobotiXX/saad/lfh_cp/bin/activate

## Execute your script
python3 LfH_main.py
