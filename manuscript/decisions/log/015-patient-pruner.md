---
status: kept
date: 2026-07-28
commits: [cde9a85]
category: training
---

# 015 — `PatientPruner` added on top of `MedianPruner`

## Context

Analysis of the v10 24h study (`results/shape_v10/shape/lead_24h/`) showed trials pruned
right at the `tf_ratio`-decay/warmup boundary with scores statistically indistinguishable
from the eventual best trial at that same epoch — the best trial itself dipped and
recovered several times before reaching its peak roughly 30 epochs later.
`optuna.pruners.MedianPruner` alone prunes on a single below-median report, with no
tolerance for a trial that is dipping-but-not-stuck.

## Change

Wrapped the existing `MedianPruner` (see [[005]]) in `optuna.pruners.PatientPruner(
median_pruner, patience=10, min_delta=0.0)` — a trial must fail `MedianPruner`'s check for
10 consecutive reports before actually being pruned, so a temporary dip no longer kills a
trial that would have recovered.

## Evidence

Reasoning-only, from the same v10 24h study analysis cited above — no isolated pruned-vs-
not-pruned trial count or a clean before/after comparison is committed to `results/`; this
is a qualitative diagnosis (the eventual best trial's own dip-and-recover trajectory) that
motivated a structural pruner change rather than a quantitative sweep.

## Decision

Kept. `interval_steps` was later changed to 5 to match `_train_model`'s smoothing window
(see [[023]]) so each `PatientPruner`/`MedianPruner` check compares fresh, largely
non-overlapping windows.

## Related

Code: `scripts/optimize.py` (pruner construction)
Other decisions: [[005]] (the underlying `MedianPruner` warmup-steps decision this extends),
[[014]] (bundled in the same commit — motivated by the same v10 24h study analysis)
Manuscript section this might feed: Methods — hyperparameter search (pruning strategy)
