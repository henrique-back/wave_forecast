#!/usr/bin/env bash
#SBATCH --job-name=optimize_v13_lead12
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /home/g1/henriqueb/Projetos/wave_forecast
source .venv/bin/activate
python scripts/optimize.py --lead 12
