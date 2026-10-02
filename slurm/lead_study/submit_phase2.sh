#!/usr/bin/env bash
# Lead-time study, phase 2 — shape_v14 transformer at 6/12/24/48/72/96 h.
# See manuscript/plans/lead_time_study.md. Submits everything at once;
# Slurm holds dependent jobs (PD, Dependency):
#
#   optimize 6, 12, 24, 48       (no dependency, 80 trials each)
#   optimize 72, 96              afterok optimize 48 — warm-started from 48 h's
#                                top trials, LONG_LEAD_TRIALS budget
#   train L                      afterok optimize L (5 seeds)
#   eval_transformer             afterok every train job
#
# An optimize job that hits --time fails its dependents: resubmit it alone
# (sbatch --export=ALL,LEAD=<L> slurm/lead_study/optimize.slurm — it resumes
# the study), then submit train/eval by hand with the new job id.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
LONG_LEAD_TRIALS="${LONG_LEAD_TRIALS:-40}"

submit() { sbatch --parsable "$@"; }

declare -A OPT TRAIN
for L in 6 12 24 48; do
    OPT[$L]=$(submit -J "opt_v14_${L}h" --export=ALL,LEAD=$L slurm/lead_study/optimize.slurm)
done
for L in 72 96; do
    OPT[$L]=$(submit -J "opt_v14_${L}h" --dependency=afterok:${OPT[48]} \
        --export=ALL,LEAD=$L,OPT_ARGS="--warm-start-from 48 --n-trials $LONG_LEAD_TRIALS" \
        slurm/lead_study/optimize.slurm)
done
for L in 6 12 24 48 72 96; do
    TRAIN[$L]=$(submit -J "train_v14_${L}h" --dependency=afterok:${OPT[$L]} \
        --export=ALL,LEAD=$L slurm/lead_study/train.slurm)
done
DEPS=$(IFS=:; echo "${TRAIN[*]}")
EVAL=$(submit -J eval_lead_curves --dependency=afterok:$DEPS slurm/lead_study/eval_transformer.slurm)

for L in 6 12 24 48 72 96; do echo "lead ${L} h: optimize ${OPT[$L]}  train ${TRAIN[$L]}"; done
echo "eval: $EVAL"
