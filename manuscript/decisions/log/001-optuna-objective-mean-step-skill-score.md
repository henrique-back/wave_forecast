---
status: kept
date: 2026-04-08
commits: [6ed4ae2]
category: objective-metric
---

# 001 — Optuna objective: teacher-forced RMSE → mean per-step Skill Score

## Context

The Optuna objective originally reported teacher-forced RMSE per trial/epoch. `seq_len`
(how many future steps the model forecasts) is itself a search-space hyperparameter, and the
persistence baseline's RMSE is not constant across `seq_len` — longer horizons are harder for
persistence too, so a raw RMSE comparison between two trials with different `seq_len` doesn't
mean what it looks like it means; it can favor a trial that "improved" mostly because
persistence got easier at its chosen `seq_len`, not because the model got better.

## Change

`nn/optimization.py::objective()` now reports `mean(per_step_SS)` — the per-step Skill Score
averaged over the forecast horizon — instead of RMSE. Early stopping was flipped to maximize
(`best_val_score` initialized to `float('-inf')`), the Optuna study direction changed to
`'maximize'`, and `lead_time` is now passed into `evaluate()` for both the val and test calls
so the persistence baseline is computed correctly at each stage.

## Evidence

This was a methodological correction, not a metric-driven ablation — there is no
before/after comparison to cite because the old objective wasn't comparable across trials in
the first place (see Context). The commit message documents the reasoning directly:
"optimize on mean per-step Skill Score instead of teacher-forced RMSE ... accounts for
varying persistence difficulty across seq_len hyperparameter configurations."

## Decision

Kept, and hardened later: this is the origin of what the codebase now calls
`weighted_mean_SS` / `Hs_SS`-style objectives, and the root `CLAUDE.md` Key Invariants
section still states the same rule verbatim — "Skill Score is not equivalent to RMSE across
Optuna trials because `seq_len` is a hyperparameter... Prefer `Hs_SS` or `weighted_mean_SS`
over raw `RMSE`."

## Related

Code: `nn/optimization.py`, `nn/evaluate.py`
Other decisions: [[009]] (a later, more targeted objective-metric experiment building on the
same "which scalar do we optimize" question)
Manuscript section this might feed: Methods — training/optimization setup, and a footnote on
why Skill Score rather than RMSE is the reported comparison metric throughout.
