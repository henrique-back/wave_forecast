#!/usr/bin/env bash
# Final retrain (5 seeds, scripts/train.py SEEDS) of one lead from the
# shape_v14 study's best trial. Submitted by submit_phase2.sh with LEAD exported.
#SBATCH --job-name=lead_train
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python scripts/train.py --experiment "${EXPERIMENT:-shape_v14}" --lead "$LEAD"
