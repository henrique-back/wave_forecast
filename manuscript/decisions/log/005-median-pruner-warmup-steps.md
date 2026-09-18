---
status: kept
date: 2026-07-13
commits: [4f6d0d0]
category: training
---

# 005 — Optuna `MedianPruner`: `n_warmup_steps` 20 → 30

## Context

Diagnosing pruning behavior in an earlier study, a large fraction of 12h-lead trials were
being pruned at exactly the warmup boundary rather than for genuinely poor performance —
noted directly in the commit as "54% of 12h trials pruned at exactly epoch 20 with
`n_warmup_steps=20`". This meant trials were being cut before scheduled-sampling and LR
decay had a chance to show whether they'd actually improve.

## Change

`scripts/optimize.py`: `optuna.pruners.MedianPruner(n_warmup_steps=20, ...)` →
`MedianPruner(n_warmup_steps=30, n_min_trials=5, interval_steps=1)`.

## Evidence

The 54% boundary-pruning statistic itself, from the same study that motivated the change —
no separate before/after Skill Score comparison was run (or is needed): the failure mode was
diagnosed directly from Optuna's pruning logs, not inferred from a downstream metric.

## Decision

Kept. Root `CLAUDE.md` documents `n_warmup_steps=30` as the current pruner config verbatim,
with the same 54% rationale still attached, so this hasn't been revisited since.

## Related

Code: `scripts/optimize.py`
Other decisions: —
Manuscript section this might feed: Methods — hyperparameter search setup (brief mention of
the pruning configuration and why the warmup was widened).
