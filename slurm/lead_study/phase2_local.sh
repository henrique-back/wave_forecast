#!/usr/bin/env bash
# Phase 2 without Slurm (a machine with its own free GPU): the same steps as
# submit_phase2.sh, run one after another. Long — days. Every step resumes
# when re-run (Optuna studies keep their trials; finished evaluations skip).
#     nohup bash slurm/lead_study/phase2_local.sh > logs/phase2.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
mkdir -p logs
export WAVE_FORECAST_ALLOW_NO_SLURM=1
LONG_LEAD_TRIALS="${LONG_LEAD_TRIALS:-40}"

for L in 6 12 24 48; do python scripts/optimize.py --lead $L; done
for L in 72 96; do python scripts/optimize.py --lead $L --warm-start-from 48 --n-trials "$LONG_LEAD_TRIALS"; done
for L in 6 12 24 48 72 96; do python scripts/train.py --experiment shape_v14 --lead $L; done
python scripts/eval_lead_curves.py --models transformer --experiment shape_v14 --device cuda --n-jobs 8
python scripts/plot_lead_curves.py
