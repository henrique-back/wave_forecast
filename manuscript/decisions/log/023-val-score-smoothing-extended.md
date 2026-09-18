---
status: kept
date: 2026-08-03
commits: [34fa691]
category: training
---

# 023 — Validation-score smoothing extended to early-stopping/checkpoint selection, not just the pruner report

## Context

Epoch-to-epoch validation score is noisy (autoregressive evaluation on a small validation
split). Originally, only the Optuna pruner report was smoothed with a trailing mean;
`best_val_score`/early-stopping/checkpoint selection used the raw per-epoch value, on the
reasoning that picking a checkpoint was lower-stakes than pruning a whole trial. This
assumption broke for the `'final_step_SS_wasserstein'` objective metric ([[021]]): its
`Shape_Wasserstein` term has enough of its own per-epoch variance (amplified by
`_FINAL_STEP_SS_WASSERSTEIN_BETA=10.0`) that an unsmoothed run locked its best checkpoint
onto an early noise spike (epoch 14) and never recognized genuinely continued improvement
in later epochs (`Shape_SS` climbing steadily through epoch 30+) as "better" — early
stopping then fired on a stale, undertrained checkpoint.

## Change

`VAL_SCORE_SMOOTHING_WINDOW=5` (`nn/optimization.py::_train_model`) — a trailing mean now
smooths the LR scheduler, early-stopping/checkpoint selection, **and** (when running as an
Optuna trial) the pruner report, from the same window, removing the failure mode described
above rather than just the pruner's narrower version of it.

## Evidence

Reasoning-only, from a specific documented failure (the epoch-14 noise-spike lock-in under
`final_step_SS_wasserstein`) — no `metrics.json`/run artifact for that specific failed run
was identified as still existing under `results/`; the account is transcribed from the code
comment. Shipped in the same commit as [[019]], [[020]], [[021]], [[022]].

## Decision

Kept. `MedianPruner`'s `interval_steps` was subsequently set to 5 to match this window (see
[[015]]'s later refinement), so each pruner check compares fresh, largely non-overlapping
smoothed windows.

## Related

Code: `nn/optimization.py::_train_model` (`VAL_SCORE_SMOOTHING_WINDOW`)
Other decisions: [[021]] (the objective metric whose failure motivated this), [[015]]
(`PatientPruner`/`interval_steps` tuned to match this window)
Manuscript section this might feed: Methods — training procedure (checkpoint selection)
