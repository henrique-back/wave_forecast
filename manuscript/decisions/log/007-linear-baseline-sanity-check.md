---
status: kept as ongoing baseline
date: 2026-08-05
commits: [fc9e9f6]
category: evaluation
---

# 007 — Per-frequency linear regression baseline vs. the transformer

## Context

After several architecture iterations, a sanity check was needed: is the transformer
actually earning its complexity over a simple linear model, and if so, at which lead times?
Without this, an improvement over the *persistence* baseline (`Skill Score`) doesn't rule out
a much cheaper model doing just as well.

## Change

Added a `linear_baseline` (per-lead-time, per-target linear regression) trained and
evaluated through the same data pipeline/splits as the transformer, under
`results/linear_baseline/{target}/lead_{N}h/`.

## Evidence

`results/comparisons/shape_v12_vs_linear_baseline/`, `shape` target:

| Lead | | `Shape_RMSE` ↓ | `Shape_SS` ↑ |
|---|---|---:|---:|
| 12h | `shape_v12` (transformer) | 2.203 | 0.121 |
| 12h | `linear_baseline` | 2.177 | 0.134 |
| 24h | `shape_v12` (transformer) | 2.443 | 0.245 |
| 24h | `linear_baseline` | 2.636 | 0.188 |

Mixed, and reported as such rather than cherry-picked: at 12h the linear baseline is
slightly *better* than the transformer on both metrics; at 24h the transformer wins clearly.
This is genuine evidence that the transformer's advantage over a simple per-frequency linear
model shows up at longer lead times, not short ones — worth stating plainly in the
manuscript rather than only reporting the 24h comparison.

## Decision

Kept as a standing baseline (not reverted — there's nothing to revert, it's a comparison
artifact) run alongside future experiments, precisely because the 12h result shows it isn't
a strawman.

## Related

Code: (baseline training script — locate under `scripts/` if promoted to a first-class CLI
command; currently produced ad hoc, see `results/linear_baseline/*/lead_*h/best_trial.txt`)
Other decisions: —
Manuscript section this might feed: Results — a linear-baseline row in the main comparison
table, with the lead-time-dependent caveat above stated explicitly rather than only quoting
the 24h number.
