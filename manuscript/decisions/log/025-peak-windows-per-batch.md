---
status: kept
date: 2026-08-17
commits: [96c704e]
category: training
---

# 025 — `SoftPeakHeightLoss` peak-window detection computed per training batch, not precomputed once per trial

## Context

`SoftPeakHeightLoss` (`utils/loss.py`) needs each peak's trough-to-trough window
(`left_idx`/`right_idx`) computed from the true spectrum before it can compute a soft peak
height. Window detection (`utils.spectral_partitioning.find_significant_peaks`/
`find_peak_windows`) is `scipy`-based, single-spectrum, non-differentiable Python looping —
the same reason `nn/evaluate.py` already treats peak detection as opt-in/evaluation-only
rather than something run every training step. The "correct" design (per
`SoftPeakHeightLoss`'s own docstring) would precompute these windows once per sample at
data-preparation time, mirroring how `freq_means`/`shape_means` are computed once in
`nn/optimization.py::_prepare_dataloaders` and threaded through training.

## Change

`nn/training_loop.py::_peak_windows_for_batch` instead recomputes peak windows for every
`(sample, step)` in every training batch, every epoch, whenever `peak_loss_weight > 0`.
This is deliberately the pragmatic choice, not the "correct" one: threading a precomputed,
dataset-level peak-window tensor through `WaveSpectralDataset`/`_prepare_dataloaders` would
touch every consumer of that Dataset's `(src, aux, y_batch)` 3-tuple across
`nn/evaluate.py` and every `scripts/*.py` that iterates a `DataLoader` directly — judged a
larger refactor than the loss-ablation study's peak-loss arm currently justifies.

## Evidence

Reasoning-only (an accepted engineering trade-off, not a performance measurement) — no
profiling data comparing per-batch vs. precomputed cost is cited or exists in `results/`.
The code comment explicitly flags this as revisit-if-profiling-shows-a-problem, not a
closed decision backed by measurement.

## Decision

Kept for now, explicitly provisional: "revisit (move to `_prepare_dataloaders`, computed
once per trial like `freq_means`/`shape_means`) if profiling shows this dominates trial
wall-clock time." This path only runs when `peak_loss_weight > 0` (zero cost otherwise), so
the trade-off is scoped to arms of the loss-ablation study that actually use this term.

## Related

Code: `nn/training_loop.py::_peak_windows_for_batch`, `utils/loss.py::SoftPeakHeightLoss`,
`utils/spectral_partitioning.py::find_peak_windows`
Other decisions: [[008]] (the underlying peak-detection algorithm), [[026]] (the loss
ablation this supports)
Manuscript section this might feed: none (implementation trade-off, not a methodological
choice) — out of scope per the parent CLAUDE.md's "write the science, not the repository"
rule.
