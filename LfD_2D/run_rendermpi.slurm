#!/bin/bash

#SBATCH --job-name=single_render

#SBATCH --mail-type=END,BEGIN,FAIL         
#SBATCH --mail-user=sghani2@gmu.edu

#SBATCH --output=logs/render_demo.%j.out
#SBATCH --error=logs/render_demo.%j.err

#SBATCH --time=3-00:00   # Total time needed for job: Days-Hours:Minutes

#SBATCH --partition=normal
#SBATCH --ntasks=720
#SBATCH --nodes=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G        # Total memory needed per task (units: K,M,G,T)

##SBATCH --array=         # seq_lens = [3,5,10] accel_jerk = [None, accel, jerk]

## Load the relevant modules needed for the job
module load gnu10 openmpi
source $HOME/miniforge/bin/activate
conda activate dyna_lflh3

## Run your program or script
## srun -m NoPack python -m mpi4py.futures render_demo_2D_mpi4.py --seq_len_idx $(( $SLURM_ARRAY_TASK_ID / 3 )) --accel_jerk $(( $SLURM_ARRAY_TASK_ID % 3 ))
srun -m NoPack python -m mpi4py.futures render_demo_2D_mpi.py 
##python3 LfD_main.py --seq_len_idx $(( $SLURM_ARRAY_TASK_ID % 4 ))  --model_type $(( $SLURM_ARRAY_TASK_ID / 4 ))
