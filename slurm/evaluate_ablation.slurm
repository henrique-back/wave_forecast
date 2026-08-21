#!/usr/bin/env bash
#SBATCH --job-name=evaluate_ablation
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /home/g1/henriqueb/Projetos/wave_forecast
source .venv/bin/activate
python scripts/evaluate_ablation_phases.py
