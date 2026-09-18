---
status: kept
date: 2026-07-28
commits: [cde9a85]
category: objective-metric
---

# 014 — Optuna objective metric: `weighted_mean_SS` → `final_step_SS`

## Context

`weighted_mean_SS` (`nn/optimization.py::_weighted_mean_ss`) exponentially downweights
later autoregressive forecast steps, which is appropriate for the aggregate reporting
metric ([[001]]) but biases trial/checkpoint *selection* toward the earlier, easier
autoregressive steps rather than the step that is actually the deliverable at a given lead
time (the intermediate steps are scaffolding to get there, not a forecast product in their
own right).

## Change

`OBJECTIVE_METRIC` (non-`hs` targets) switched from `'weighted_mean_SS'` to
`'final_step_SS'` — Skill Score at the last forecast step only
(`nn/optimization.py::_compute_val_score`). This changes what "best" means for a given
trial's checkpoint and for cross-trial comparison, so pre-change trials are not comparable
to post-change ones.

## Evidence

Reasoning-only: the change is a selection-criterion correctness argument (which step should
determine "best"), not something backed by an isolated before/after metrics comparison —
no `metrics.json` isolates this change from the `PatientPruner` addition ([[015]]) bundled
in the same commit. `results/shape_v10/` is the first study run with `final_step_SS` as the
objective.

## Decision

Kept. Every study since (v10 onward, including [[009]]'s later `peak_fidelity_SS` and
[[021]]'s `final_step_SS_wasserstein`) is built on the same "score the actual deliverable
step" principle established here.

## Related

Code: `nn/optimization.py::_compute_val_score`
Other decisions: [[001]] (aggregate reporting metric — different question, "how do we
report performance across the whole horizon" vs. this entry's "what does Optuna optimize
for"), [[009]] (`peak_fidelity_SS`, a further evolution for the loss-ablation study),
[[015]] (bundled in the same commit — the v10 24h study analysis that motivated this metric
switch also motivated adding `PatientPruner`)
Manuscript section this might feed: Methods — hyperparameter search / objective definition
