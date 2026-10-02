#!/usr/bin/env bash
# Rolls every shape_v14 seed checkpoint out to 96 h, adds it to the
# phase-1 statistics, and redraws the figures and tables.
#SBATCH --job-name=lead_eval
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python scripts/eval_lead_curves.py --models transformer --experiment "${EXPERIMENT:-shape_v14}" \
    --device cuda --n-jobs 8
python scripts/plot_lead_curves.py
