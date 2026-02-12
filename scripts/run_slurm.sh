#!/bin/bash
#SBATCH --job-name=fedranklite
#SBATCH --output=results/slurm_%j.out
#SBATCH --error=results/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

CONFIG=${1:-"configs/experiments/lite_noniid_deadline.yaml"}

source ~/.bashrc
conda activate fedranklite  # ajuste para seu env

python scripts/run.py --config "$CONFIG"
