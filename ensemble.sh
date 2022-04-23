#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH -o out.log
#SBATCH -e err.log
module load nvidia/cuda/10.0
module load pytorch/1.0_python3.7_gpu
python src/ensemble.py \
  --checkpoints checkpoints/pt6_64 checkpoints/pt7_64 checkpoints/pt9_64 checkpoints/pt9_16 checkpoints/pt10_64\
  --output_dir ensemble_results.txt