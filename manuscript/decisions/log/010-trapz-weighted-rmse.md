---
status: kept
date: 2026-07-16
commits: [02aedaa]
category: training
---

# 010 — RMSE (training loss + evaluation metrics) frequency-weighted by `utils.trapz_weights` instead of a flat mean

## Context

`density`/`shape` targets' RMSE — both the training loss (`utils/loss.py::RMSELoss`) and
the RMSE/CC/Bias/R2/SS metric family in `nn/evaluate.py` + `nn/spectrum_eval.py` — was a
flat elementwise mean over the 47-bin log-spaced frequency grid. The grid is dense near
0.02 Hz and coarse near 0.485 Hz, so a flat mean over bins over-weights the dense
low-frequency region relative to its actual share of the physical spectrum (an integral
over frequency, not a count of bins).

## Change

Introduced `utils.trapz_weights(freqs)` (per-bin trapezoidal weight, summing to 1) and
threaded it through `RMSELoss.forward`'s `weights` argument and every place in
`nn/evaluate.py`/`nn/spectrum_eval.py` that collapses the frequency axis, so the
loss/metric approximates ∫(pred−true)² df rather than a bin-count mean. `STUDY_VERSION`
bumped v7→v8 in `scripts/optimize.py` since this changes what the model optimizes for and
how trials are scored — v7 trials are not comparable to v8 trials.

## Evidence

No isolated before/after metrics comparison exists for this change alone — it is a
methodological correction (the flat-mean weighting was judged wrong on physical grounds,
not chosen/rejected by an ablation), the same category as decision [[001]]'s own framing.
`results/RESEARCH_LOG.md`'s `shape_v8` row is the first study run under this weighting;
`baseline_v2`/earlier rows predate it and are not a controlled comparison against it (the
architecture and objective metric also differ across that gap — see [[002]]).

## Decision

Kept. The physical argument (RMSE should track integrated error, not per-bin-count error,
on a non-uniform grid) stands independent of any ablation result, and every study since v8
has used it.

## Related

Code: `utils/compute_hs.py::trapz_weights`, `utils/loss.py::RMSELoss`, `nn/evaluate.py`,
`nn/spectrum_eval.py`
Other decisions: [[001]] (same "methodological correction, not a clean ablation" framing)
Manuscript section this might feed: Methods — loss function / evaluation metrics
