---
status: kept
date: 2026-07-31
commits: [bb5dfdc]
category: architecture
---

# 016 — `density`/`shape` targets switched to predicting log-spectral-energy directly

## Context

`density`/`shape` targets previously predicted a Softplus-activated non-negative linear
value in physical (or unit-area shape) space, with RMSE computed in that same physical
space. Non-negativity was enforced architecturally (the Softplus), and negative energy
values were separately clamped at inference time (see the earlier padding-mode/clamp
commit that introduced that Softplus, predating this entry).

## Change

The decoder's output head switched to a plain `Linear` layer predicting
log-spectral-energy (`log E(f)` for `density`, `log(E(f)/m0)` for `shape`) directly —
`utils/log_transform.py::to_log_space` converts the physical target to log-space
(floored, per that module) immediately after load, before it's used to build the decoder
input, the scheduled-sampling self-feedback loop, or the loss target. The training loss for
these two targets switched from frequency-weighted RMSE to frequency-weighted plain MSE,
computed in log-space (`utils/loss.py::RMSELoss`'s `squared=True` path). Non-negativity of
the physical shape now comes from `exp()` at inference/metric time instead of the
architectural Softplus constraint.

Note on versioning: the code comment documenting this (`scripts/optimize.py`'s
`STUDY_VERSION` history) states it was written when the change shipped, but `STUDY_VERSION`
itself was left at `"v11"` — no `optuna_study_v12.db` or `results/shape_v12/` were produced
under a "v12" label for this specific change; `shape_v11`'s actual results already reflect
this log-space behaviour. The `v12` label was reused later for a separate, unrelated
change (see [[019]]–[[023]]). This entry documents the log-space switch itself, dated to
the commit that shipped it (`bb5dfdc`), independent of the `STUDY_VERSION` labeling
confusion.

## Evidence

`git show bb5dfdc` (message: "v11 modifications and partial results") contains the
`nn/training_loop.py`/`nn/transformer.py`/`nn/evaluate.py` diff implementing the switch.
No isolated physical-space-Softplus-RMSE vs. log-space-MSE comparison exists in
`results/` — `shape_v11`'s results are the first (and only) study run this way; nothing
comparable was re-run under the old regime for a controlled comparison.

## Decision

Kept. Two effects were flagged when the change shipped and should be kept in mind when
interpreting `shape_v11`/later results: (1) `lr`'s search range was still bracketing
`shape_v9`'s best trials under the OLD physical-space-RMSE regime, not re-validated for the
new log-space MSE; (2) `OBJECTIVE_METRIC` choices derived from `per_step_SS`/`overall_SS`
are computed in log-space for `density`/`shape` and are not comparable to pre-this-change
runs, while `Hs_SS`, `Tm02_RMSE`, and `Shape_RMSE` are `exp()`'d back to physical units
internally (`nn/evaluate.py`) and remain the fair basis for such comparisons.

## Related

Code: `nn/transformer.py` (predictor construction), `nn/training_loop.py`,
`nn/evaluate.py`, `utils/log_transform.py::to_log_space`
Other decisions: [[019]]–[[023]] (later changes reusing the "v12" `STUDY_VERSION` label)
Manuscript section this might feed: Methods — model output representation / loss function
