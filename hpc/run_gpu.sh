#!/bin/bash
# Usage:  module swap cluster/accelgor && sbatch $VSC_DATA/cusp-mace/run_gpu.sh <script.py> [args...]
# Output: slurm-<jobid>.out in the directory you submit from.
#SBATCH --job-name=mace
#SBATCH --partition=accelgor
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
source $VSC_DATA/cusp-mace/env.sh
echo "node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python "$@"
