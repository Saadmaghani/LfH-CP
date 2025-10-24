#!/bin/bash

#SBATCH --job-name=LfD_main_simple
#SBATCH --partition=gpuq        
#SBATCH --qos=gpu                      
#SBATCH --gres=gpu:3g.40gb:1        

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30gb                  
#SBATCH --time=3-00:00:00             

#SBATCH --output=logs/LfD_main.%j.out
#SBATCH --error=logs/LfD_main.%j.err

#SBATCH --mail-user=sghani2@gmu.edu
#SBATCH --mail-type=FAIL,REQUEUE,END,BEGIN

##SBATCH --array=12

# Activate environment
source $HOME/miniforge/bin/activate
conda activate dyna_lflh3

# running LfD_main on seq_length = {3,5,10,20} cmd_predictions = {1, 3, 5, 10}

# Execute script with array index determining parameters
echo "Running on $(hostname)"
# python -u LfD_main.py --seq_len_idx $(( $SLURM_ARRAY_TASK_ID % 5 ))  --cmd_predictions_idx $(( $SLURM_ARRAY_TASK_ID / 5 ))
python -u LfD_main.py --seq_len_idx 0  --cmd_predictions_idx 3

