#!/bin/bash
# Usage:  module swap cluster/joltik && sbatch $VSC_DATA/mlff-gcmc/run_gpu_joltik.sh <script.py> [args...]
#SBATCH --job-name=mace
#SBATCH --partition=joltik
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
module swap cluster/joltik 2>/dev/null
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0 ASE/3.24.0-gfbf-2024a
source $VSC_DATA/mlff-gcmc/venv/bin/activate
echo "node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python "$@"
