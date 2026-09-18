---
status: kept
date: 2026-08-03
commits: [34fa691]
category: training
---

# 022 — Linear LR warmup added (`WARMUP_EPOCHS=5`)

## Context

The transformer had no LR warmup and trained AdamW at the sampled `lr` from epoch 0.
Optuna hyperparameter-importance analysis of `shape_v11`/`shape_v12` found `lr` dominating
importance (0.23–0.69 across lead times) while only weakly correlating with score
(0.23–0.47) — the best and worst trials at every lead time drew `lr` from almost the same
range, which is the signature of noisy/unstable early training rather than a clean optimum
TPE can exploit.

## Change

Added a linear LR warmup ramping up to the sampled `lr` over the first `WARMUP_EPOCHS=5`
epochs (`nn/optimization.py::_train_model`). `ReduceLROnPlateau` only starts stepping once
warmup ends, so the ramp itself is never mistaken for a plateau.

## Evidence

`results/shape_v11/shape/lead_{6,12}h/param_importances.html` and the equivalent
`shape_v12` files are cited by the code comment as the source of the 0.23–0.69
importance / 0.23–0.47 correlation figures; these files exist under `results/shape_v11/`.
No isolated with-warmup vs. without-warmup comparison exists — this was diagnosed from
hyperparameter-importance statistics, not an ablation, and shipped in the same commit as
[[019]], [[020]], [[021]], [[023]].

## Decision

Kept. The instability signature in `lr`'s importance/correlation profile is a standard,
well-understood transformer training pathology that warmup is the standard fix for; no
regression was observed in subsequent studies.

## Related

Code: `nn/optimization.py::_train_model` (`WARMUP_EPOCHS`)
Manuscript section this might feed: Methods — training procedure (LR schedule)
