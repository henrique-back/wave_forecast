#!/usr/bin/env bash
# Lead-time study, phase 1 — ridge AR, persistence and GEFSv12 (no GPU, no
# Slurm). See manuscript/plans/lead_time_study.md. Run from anywhere:
#     bash slurm/lead_study/phase1_ar.sh
# Each step is resumable; re-running skips finished work except the AR grid
# (about 20 s per lead) and the plots.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
LEADS="6 12 24 48 72 96"
N_JOBS="${N_JOBS:-8}"

echo "== 1/5 tests"
WAVE_FORECAST_ALLOW_NO_SLURM=1 pytest -q tests/test_peak_records.py

echo "== 2/5 AR order selection on val (final-step peak_fidelity) and test fit"
python scripts/optimize_linear_baseline.py --name linear_baseline_pf --objective peak_fidelity --leads $LEADS
python scripts/train_linear_baseline.py --name linear_baseline_pf --leads $LEADS

echo "== 3/5 GEFSv12 reforecast to +96 h (internet; resumable)"
python scripts/fetch_gefs_reforecast.py

echo "== 4/5 forecasts and peak statistics"
python scripts/eval_lead_curves.py --models persistence ar gefs --ar-name linear_baseline_pf \
    --leads $LEADS --n-jobs "$N_JOBS"

echo "== 5/5 figures and tables"
python scripts/plot_lead_curves.py
echo "Done: results/comparisons/lead_curves/figures/"
