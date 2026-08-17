#!/usr/bin/env bash
# Submits all 5 loss-ablation phases (see scripts/ablate_loss.py) to Slurm in
# one shot, respecting the dependency graph: 'wasserstein' and 'peak' each
# need 'kl' phase's winning kl_loss_weight (read from its current_best.txt),
# and 'combined' needs both 'wasserstein' and 'peak' finished first.
# 'baseline' depends on nothing.
#
#   baseline  (no dependency)
#   kl        (no dependency)
#   wasserstein  --dependency=afterok:<kl job id>
#   peak         --dependency=afterok:<kl job id>
#   combined     --dependency=afterok:<wasserstein job id>:<peak job id>
#
# afterok (not just 'after'): a dependent phase only starts if the phase it
# reads current_best.txt from actually finished successfully — no point
# burning GPU time reading a file a failed run never wrote.
#
# All 5 are submitted immediately; Slurm holds the dependent ones in state
# PD (Dependency) until their prerequisite completes — this is what
# "schedule all phases" means here, not that they all start running now.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

submit() {
    # sbatch prints "Submitted batch job <N>" — grab just <N>.
    sbatch "$@" | awk '{print $NF}'
}

echo "Submitting baseline and kl (no dependencies)..."
BASELINE_ID=$(submit slurm/ablate_baseline.slurm)
KL_ID=$(submit slurm/ablate_kl.slurm)
echo "  baseline: job $BASELINE_ID"
echo "  kl:       job $KL_ID"

echo "Submitting wasserstein and peak (depend on kl=$KL_ID)..."
WASSERSTEIN_ID=$(submit --dependency=afterok:"$KL_ID" slurm/ablate_wasserstein.slurm)
PEAK_ID=$(submit --dependency=afterok:"$KL_ID" slurm/ablate_peak.slurm)
echo "  wasserstein: job $WASSERSTEIN_ID"
echo "  peak:        job $PEAK_ID"

echo "Submitting combined (depends on wasserstein=$WASSERSTEIN_ID and peak=$PEAK_ID)..."
COMBINED_ID=$(submit --dependency=afterok:"$WASSERSTEIN_ID":"$PEAK_ID" slurm/ablate_combined.slurm)
echo "  combined: job $COMBINED_ID"

echo
echo "All 5 phases scheduled. Monitor with: squeue -u $USER"
echo "Cancel one with: scancel <job id>   |   cancel everything above: scancel $BASELINE_ID $KL_ID $WASSERSTEIN_ID $PEAK_ID $COMBINED_ID"
