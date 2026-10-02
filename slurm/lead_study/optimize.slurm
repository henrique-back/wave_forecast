#!/usr/bin/env bash
# One lead of the shape_v14 search. Submitted by submit_phase2.sh with
# LEAD (and optionally OPT_ARGS) exported; resubmitting resumes the study.
#SBATCH --job-name=lead_optimize
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python scripts/optimize.py --lead "$LEAD" ${OPT_ARGS:-}
