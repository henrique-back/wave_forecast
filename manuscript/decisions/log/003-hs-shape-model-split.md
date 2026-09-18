---
status: kept
date: 2026-07-07
commits: [98683cc, 608cdd7]
category: architecture
---

# 003 — Two separate `hs`+`shape` models recombined at inference, vs. one monolithic `density` model

## Context

A single `density`-target model has to jointly get the spectrum's total energy (magnitude)
and its shape right in one prediction, and was observed to underestimate spectral peaks and
over-smooth the high-frequency tail. The hypothesis: decouple the (typically easier)
magnitude-forecasting problem from the (typically harder) shape-forecasting problem by
training two separate models — one predicting scalar `hs` (Hs = 4√m₀), one predicting the
unit-area normalized shape `E(f)/m₀` — and recombine at inference:
`E_pred(f,t) = shape_pred(f,t) * ((hs_pred(t)/4)^2)`.

## Change

Added `target='hs'` and `target='shape'` (via `compute_shape`, `utils/compute_hs.py`) as
first-class Optuna/train targets alongside `density`. `scripts/infer.py --target combined`
and `scripts/compare_versions.py --experiment name:combined` load a matching `hs` checkpoint
and `shape` checkpoint (same experiment/lead) and do the recombination. See root `CLAUDE.md`
"Shape/magnitude model split" for the full mechanism.

## Evidence

`results/comparisons/weightedmeanSS_conv_freqemb_v3_vs_hs_shape_v5_vs_hs_shape_v6/lead_6h/metrics.json`,
lead 6h:

| | `overall_SS` ↑ | `Hs_RMSE` ↓ | `Shape_RMSE` ↓ |
|---|---:|---:|---:|
| `weightedmeanSS_conv_freqemb_v3` (monolithic `density`) | 0.1406 | 0.1931 | 2.108 |
| `hs_shape_v5` (combined) | 0.1676 | 0.1867 | 2.016 |
| `hs_shape_v6` (combined) | 0.1944 | 0.1716 | 1.980 |

The combined split beats the monolithic model on all three metrics at this lead time, and
`hs_shape_v6` (a later iteration with increased patience / pruner warmup, see [[005]]) is
the best of the three. Not a perfectly isolated ablation either — `hs_shape_v5`/`v6` are
later studies than `v3` and picked up other incremental changes in the interim — but the
direction is consistent with the stated hypothesis (monolithic model over-smoothing) and
holds across two independent `hs_shape` runs, not just one.

## Decision

Kept. `hs_shape_v5`/`v6` became the reference "combined" pipeline going forward
(`scripts/infer.py`, `scripts/compare_versions.py` both special-case `--target combined`).

## Related

Code: `utils/compute_hs.py::compute_shape`, `scripts/infer.py`, `scripts/compare_versions.py`
Other decisions: [[002]], [[005]]
Manuscript section this might feed: Methods — model targets/outputs; Results — density vs.
combined comparison table (the numbers above are safe to cite directly, with the same-lead
caveat about later incidental changes in v5/v6 noted).
