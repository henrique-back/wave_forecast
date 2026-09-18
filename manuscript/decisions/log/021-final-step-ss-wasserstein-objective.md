---
status: kept
date: 2026-08-03
commits: [34fa691]
category: objective-metric
---

# 021 — `final_step_SS_wasserstein` objective metric + fixed blend constant `BETA=10.0`

## Context

Once training could include the auxiliary Wasserstein loss term ([[020]]), selecting the
best epoch/trial by plain `final_step_SS` ([[014]]) would be blind to
`Shape_Wasserstein` — risking silently discarding a better-separated-peaks checkpoint in
favor of one that is marginally better on a metric known to dilute exactly that property
(the same aggregate-metric-dilutes-peak-behaviour problem this project repeatedly
encounters, e.g. [[009]]).

## Change

Added `'final_step_SS_wasserstein'` as an `OBJECTIVE_METRIC` choice
(`nn/optimization.py::_compute_val_score`): `final_step_SS − BETA × Shape_Wasserstein`.
The blend weight `_FINAL_STEP_SS_WASSERSTEIN_BETA = 10.0` is a **fixed constant**,
deliberately not the trial's own tunable `wasserstein_loss_weight` — using the trial's own
weight here would make cross-trial comparison unfair (trials sampling a large
`wasserstein_loss_weight` would get a structurally different scoring scale purely from that
hyperparameter choice).

## Evidence

`BETA=10.0` was chosen by rough order-of-magnitude matching against observed test-set
ranges from [[020]]'s manual validation, explicitly **not a precisely fit constant**: per
the code comment, `final_step_SS`-family metrics sit around 0.1–0.2 in this problem (e.g.
best `val_weighted_mean_SS` was 0.17–0.18 across the manually-tested configs), while
`Shape_Wasserstein` sits around 0.011–0.014 for well-trained models — `BETA=10` puts a
typical `Shape_Wasserstein` contribution (~0.1–0.14) on a comparable scale to a typical SS
value. No committed study data was used to fit this value; the comment itself flags
"revisit once real study data exists."

## Decision

Kept as an open/provisional choice — the constant is a placeholder scaled to be
"non-dominated," not validated by a search or sensitivity analysis. This metric was itself
later superseded by `'peak_fidelity_SS'` for the loss-ablation study once training stopped
optimizing RMSE at all (`base_loss_weight=0`, see [[026]]) — `final_step_SS_wasserstein` is
an RMSE transform, so it stopped being a fair selection criterion in that regime (see
[[009]]'s same reasoning). Retained as a valid choice for runs that do still train on RMSE
plus a Wasserstein term.

## Related

Code: `nn/optimization.py::_compute_val_score`,
`nn/optimization.py::_FINAL_STEP_SS_WASSERSTEIN_BETA`
Other decisions: [[014]] (base `final_step_SS`), [[020]] (the loss term this metric tracks),
[[009]] (`peak_fidelity_SS`, the metric that superseded this one for the RMSE-free loss
ablation), bundled in the same commit as [[019]], [[022]], [[023]]
Manuscript section this might feed: Methods — hyperparameter search / objective definition
(flag the unvalidated constant if quoted).
