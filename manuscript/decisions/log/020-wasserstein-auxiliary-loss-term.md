---
status: kept
date: 2026-08-03
commits: [34fa691]
category: training
---

# 020 — Auxiliary `SpectralWassersteinLoss` term added to training (`wasserstein_loss_weight`)

## Context

A whole-spectrum frequency-weighted loss (RMSE/MSE, see [[010]]/[[016]]) gives multimodal
(double/triple-peaked) sea states no special treatment: it tends to blur two distinct peaks
into one smoothed hump rather than preserving their separation, since a pointwise loss
penalizes a slightly-shifted sharp peak almost as harshly as a fully displaced one.

## Change

Added `wasserstein_loss_weight` as a new tunable hyperparameter
(`nn/optimization.py::objective`): when > 0, `wasserstein_loss_weight *
SpectralWassersteinLoss(y_pred, y_batch, freqs)` is added to the main per-bin loss for
`target == 'shape'` (`nn/training_loop.py::train_one_epoch`). At the time this was added,
`SpectralWassersteinLoss` computed the 1-D Wasserstein-1 (earth-mover) distance between
predicted and true spectra via the exact CDF-L1 shortcut (later switched to Wasserstein-2 —
see [[024]]).

## Evidence

**Manually swept, not committed to `results/`**: weights 50 and 150 were compared against a
0-weight control (reusing `shape_v11`'s exact other hyperparameters). Per the code comment
(`scripts/optimize.py`'s `STUDY_VERSION` v12 entry): every metric improved monotonically at
weight 150 — `Shape_RMSE` 2.244→2.083, `Shape_SS` 0.107→0.171, `Peak_Separation_Recall`
0.580→0.617 — and the improvement was confirmed visually sharper on known-multimodal test
samples, not just numerically better. No `metrics.json` file for this specific manual sweep
was found under `results/` — the numbers above are transcribed from the code comment
itself, per this log's disclosure rule for non-committed evidence.

Note: these weights (50/150) and the 10–400 search range initially wired into
`objective()` were tuned under the **old Wasserstein-1** metric's scale. Wasserstein-2's
values are on a different numeric scale (see [[024]]) — any trial completed before
2026-08-17 used a different, incomparable `wasserstein_loss_weight` definition than trials
completed after it.

## Decision

Kept, wired into the search space as a tunable hyperparameter. The range was later replaced
wholesale under the current W2 metric ([[027]]).

## Related

Code: `utils/loss.py::SpectralWassersteinLoss`, `nn/training_loop.py::train_one_epoch`
Other decisions: [[019]] (bundled in the same commit), [[024]] (W1→W2 switch that
invalidated this entry's tuned weights), [[026]] (composite-loss ablation that re-tuned this
term under W2), [[027]] (search-space range replacement)
Manuscript section this might feed: Methods — loss function (multimodal/peak-separation
term). See `manuscript/decisions/wasserstein_kl_justification.tex` for the full
citation-backed argument — **note that .tex doc currently only argues for W1; see [[024]]**.
