#!/usr/bin/env bash
# Submits all 6 loss-ablation phases (see scripts/ablate_loss.py) to Slurm in
# one shot, respecting the dependency graph: 'wasserstein' and 'peak' each
# need 'kl' phase's winning kl_loss_weight (read from its current_best.txt),
# and 'combined' needs both 'wasserstein' and 'peak' finished first.
# 'baseline' and 'wasserstein_only' depend on nothing — the latter exists
# precisely to test whether Wasserstein needs KL underneath it at all, so it
# must not read kl's winner. (There is no 'peak_only' counterpart: the peak
# term cannot stand alone — see manuscript/decisions/log/032.)
#
#   baseline          (no dependency)
#   kl                (no dependency)
#   wasserstein_only  (no dependency)
#   wasserstein       --dependency=afterok:<kl job id>
#   peak              --dependency=afterok:<kl job id>
#   combined          --dependency=afterok:<wasserstein job id>:<peak job id>
#   evaluate          --dependency=afterok:<combined job id>
#
# afterok (not just 'after'): a dependent phase only starts if the phase it
# reads current_best.txt from actually finished successfully — no point
# burning GPU time reading a file a failed run never wrote.
#
# All are submitted immediately; Slurm holds the dependent ones in state
# PD (Dependency) until their prerequisite completes — this is what
# "schedule all phases" means here, not that they all start running now.
# Note the QOS caps concurrent jobs, so in practice these run one at a time
# regardless of the graph: budget ~6-8h per phase, i.e. ~2-2.5 days total.
#
# The final evaluate job recomputes every phase's panel on the held-out TEST
# split (scripts/evaluate_ablation_phases.py) so the comparison table isn't
# reading the same validation numbers that selected each winner. Then:
#   python scripts/compare_ablation_phases.py \
#       --study-version lossablation_v3 --out results/lossablation_comparison_v3.md

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

submit() {
    # sbatch prints "Submitted batch job <N>" — grab just <N>.
    sbatch "$@" | awk '{print $NF}'
}

echo "Submitting baseline, kl, wasserstein_only (no dependencies)..."
BASELINE_ID=$(submit slurm/ablate_baseline.slurm)
KL_ID=$(submit slurm/ablate_kl.slurm)
WASSERSTEIN_ONLY_ID=$(submit slurm/ablate_wasserstein_only.slurm)
echo "  baseline:         job $BASELINE_ID"
echo "  kl:               job $KL_ID"
echo "  wasserstein_only: job $WASSERSTEIN_ONLY_ID"

echo "Submitting wasserstein and peak (depend on kl=$KL_ID)..."
WASSERSTEIN_ID=$(submit --dependency=afterok:"$KL_ID" slurm/ablate_wasserstein.slurm)
PEAK_ID=$(submit --dependency=afterok:"$KL_ID" slurm/ablate_peak.slurm)
echo "  wasserstein: job $WASSERSTEIN_ID"
echo "  peak:        job $PEAK_ID"

echo "Submitting combined (depends on wasserstein=$WASSERSTEIN_ID and peak=$PEAK_ID)..."
COMBINED_ID=$(submit --dependency=afterok:"$WASSERSTEIN_ID":"$PEAK_ID" slurm/ablate_combined.slurm)
echo "  combined: job $COMBINED_ID"

echo "Submitting test-set evaluation (depends on combined=$COMBINED_ID)..."
EVALUATE_ID=$(submit --dependency=afterok:"$COMBINED_ID" slurm/evaluate_ablation.slurm)
echo "  evaluate: job $EVALUATE_ID"

ALL_IDS="$BASELINE_ID $KL_ID $WASSERSTEIN_ONLY_ID $WASSERSTEIN_ID $PEAK_ID $COMBINED_ID $EVALUATE_ID"

echo
echo "All 6 phases + evaluation scheduled. Monitor with: squeue -u $USER"
echo "Cancel one with: scancel <job id>   |   cancel everything above: scancel $ALL_IDS"
