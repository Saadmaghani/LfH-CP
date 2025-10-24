#!/bin/bash
#SBATCH --job-name=ddp_lfd
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=450GB
#SBATCH --time=3-00:00
#SBATCH --output=logs/ddp_lfd.%j.out
#SBATCH --error=logs/ddp_lfd.%j.err
#SBATCH --array=4

source $HOME/miniforge/bin/activate
conda activate dyna_lflh3

export MASTER_PORT=${MASTER_PORT:-12355}
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)

## running LfD_main on seq_length = {3,5,10,20} cmd_predictions = {1, 3, 5, 10}
torchrun --nproc_per_node=4 --nnodes=1 LfD_main_DDP.py --seq_len_idx $(( $SLURM_ARRAY_TASK_ID % 4 ))  --cmd_predictions_idx $(( $SLURM_ARRAY_TASK_ID / 4 ))
