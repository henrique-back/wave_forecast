---
status: exploratory — open
date: 2026-08-21
commits: [a89b19c, 5586453]
category: objective-metric
---

# 009 — `peak_fidelity_SS` as an Optuna objective metric

## Context

Whole-spectrum metrics like `weighted_mean_SS`/`Shape_RMSE` (see [[001]]) can be dominated by
the bulk of the spectrum and dilute exactly the failure mode `peak_modality_metrics` ([[008]])
was built to surface — misplaced or missing peaks in multimodal sea states. The question:
would optimizing Optuna trials directly against a peak-fidelity-based Skill Score, instead of
the whole-spectrum objective, produce a model that's actually better where it matters
(peak height/separation), even if whole-spectrum RMSE looks worse?

## Change

Added `peak_fidelity_SS` as a trial-selectable objective (built on the peak-detection work in
[[008]]), used for the `shape_v13` study (and promoted the composite KL/Wasserstein/peak loss
into that same search space per commit `5586453`).

## Evidence

`results/comparisons/shape_v13_vs_shape_v12_vs_linear_baseline_peak_fidelity/lead_12h/peak_fidelity_metrics.json`:

| | `peak_fidelity_SS` ↑ | `Peak_Separation_Recall` (windsea/swell) ↑ | `Shape_RMSE` ↓ | `Shape_SS` ↑ |
|---|---:|---:|---:|---:|
| `shape_v13` (peak_fidelity-optimized) | 0.559 | 0.931 / 0.968 | 2.766 | −0.103 |
| `shape_v12` (weighted_mean_SS-optimized) | 0.104 | 0.347 / 0.569 | 2.203 | 0.121 |
| `linear_baseline` | 0.428 | 0.693 / 0.839 | 2.177 | 0.134 |

The trade-off is stark and real: `shape_v13` dramatically wins on peak fidelity and peak
separation recall, but its whole-spectrum `Shape_RMSE`/`Shape_SS` is *worse than the linear
baseline* (negative `Shape_SS` — worse than persistence). This is a genuine finding, not a
clean win either direction.

## Decision

Still open. `peak_fidelity_SS` is not (yet) listed among the documented `OBJECTIVE_METRIC`
values in the root `CLAUDE.md` (`weighted_mean_SS`, `overall_SS`, `Hs_SS`, `RMSE`,
`Hs_RMSE`, `Tm02_RMSE`, `Shape_RMSE`, `SI_mean`) — treat it as experimental until a follow-up
resolves whether the whole-spectrum regression is an acceptable cost for peak fidelity, or
whether a combined/weighted objective is needed instead of picking one or the other.

## Related

Code: (peak-fidelity objective computation — locate current call site in `nn/optimization.py`
before citing further; wasn't present in root `CLAUDE.md`'s objective-metric list as of this
writing)
Other decisions: [[001]], [[008]]
Manuscript section this might feed: Discussion — a "whole-spectrum vs. peak-level accuracy is
a real trade-off, not yet resolved" paragraph, using the table above directly. Do not present
`shape_v13` as a straightforward improvement in Results without also reporting the
`Shape_SS` regression.
