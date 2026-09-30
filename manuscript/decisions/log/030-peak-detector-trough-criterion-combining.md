---
status: kept
date: 2026-09-30
commits: []
category: evaluation
---

# 030 — Peak detector: criterion 3 measured to the trough, spurious partitions combined; physical labels in `evaluate()`

## Context

While following up [[029]], a check of `utils/spectral_partitioning.py::find_significant_peaks`
against Portilla et al. (2009, §2b.2) found two departures from the criteria that [[008]],
CLAUDE.md and `02_methods.tex` describe:

- **Criterion 3 counted bins to the neighbouring maximum, not to the trough.** Portilla rejects
  "partitions having few spectral bins before or after the peak (i.e., less than 2 bins)". A
  partition is bounded by the minima between peaks. The code measured `idx - raw_peaks[i-1]`.
  Two local maxima are always at least 2 bins apart, so with `min_bins = 2` the test never
  rejected an interior peak. On the 32012 test split it fired 138 times in about 24,000 local
  maxima, all at the grid edges.
- **Spurious partitions were dropped, not combined.** Portilla's is a "partitioning–combining"
  scheme. The code discarded rejected peaks, and each survivor's window still stopped at the
  trough to its nearest *raw* maximum. So windows were fragments, and criterion 2 measured a
  fragment's energy.

Measuring criterion 3 to the trough *without* combining is worse: a dominant peak with a 1-bin
ripple two bins away fails criterion 3 together with the ripple. On the test split, 127 spectra
then had no peak at all.

[[028]] had already recorded the symptom without the cause: 3.69 true against 7.88 predicted
peaks per spectrum, with recall inflated because a true peak often has a predicted one nearby by
chance.

The same change closes the open item of [[029]]: `evaluate()` labelled wind sea/swell on the
unit-area shape.

## Change

- `utils/spectral_partitioning.py::_combined_partitions`, shared by `find_significant_peaks` and
  `find_peak_windows`:
  - Criterion 3 is `idx - lo < min_bins or hi - idx < min_bins`, where `lo`/`hi` are the
    partition's troughs.
  - Combining loop, repeated until no peak fails:
    1. Partition at the minima between the current peaks and check the four criteria.
    2. Take the failing peak with the lowest S(f_p) and merge it with the neighbour across its
       shallower trough.
    3. The merged pair keeps the higher peak. A failing peak with no neighbour is dropped.
  - The windows now tile the grid. Portilla does not specify an order; this one is ours.
- `nn/optimization.py::_prepare_dataloaders` attaches the true m₀ per (sample, step) to the
  val/test `WaveSpectralDataset` as `m0_true` for the `shape` target.
- `nn/evaluate.py` scales both final-step shapes by `m0_true` before the peak panel, the same
  rule as `nn/spectrum_eval.py::compute_shape_final_metrics(m0_true=...)`.
- `scripts/compare_physical_baseline.py` passes `m0_true` through its `Subset`.
- The detector is shared, so the `SoftPeakHeightLoss` training windows
  (`nn/training_loop.py::_peak_windows_for_batch`) change too, for any run from this commit on.
  shape_v13 and the [[026]] ablation were trained with the old windows.
- Tests:
  - `tests/test_spectral_partitioning.py::TestCombinedPartitions`: the ripple case, tiling, and
    survival of the global maximum. The first two fail on the old detector.
  - `tests/test_shape_final_metrics.py::test_matches_evaluate_with_physical_labels`.
  - `tests/test_optimization.py::TestShapeLoaderCarriesM0`.

## Evidence

All numbers come from `results/comparisons/peak_detector_030/`. The old detector is kept there
as `old_partitioning.py`, a copy of the file at commit 4534f47.

**Detector on the truth.** `truth_detector_stats.txt`, buoy 32012 test split, 2,632 raw hourly
spectra:

| Detector | Peaks per spectrum | ≥ 2 peaks | > 4 peaks (loss cap) | Median window | Global max kept |
|---|---|---|---|---|---|
| Old | 4.21 | 99.4% | 39.5% | 5 bins | 2,632 |
| New | 2.53 | 90.4% | 1.3% | 16 bins | 2,629 |

**shape_v13 re-scored** (`rescore_shape_v13.json`):
- Checkpoints: `results/shape_v13/shape/lead_{12,24,48}h/best_model.pt`, the local copies.
- Method: autoregressive inference, final step. Each detector and label combination is applied
  to the same forecast arrays.
- Columns: *old* is the old detector with shape-space labels, which is what was reported and what
  selected v13. *+labels* is the old detector with physical labels, the [[029]] fix alone. *new*
  applies both fixes.

| Lead | Split | PF old | PF +labels | PF new | Pred. peaks old → new | True peaks new | Recall old → new | Recall wind sea / swell (new) |
|---|---|---|---|---|---|---|---|---|
| 12 h | val | 0.576 | 0.573 | −0.052 | 7.88 → 1.24 | 2.26 | 0.966 → 0.406 | 0.192 / 0.568 |
| 12 h | test | 0.560 | 0.558 | −0.095 | 7.89 → 1.31 | 2.53 | 0.949 → 0.356 | 0.189 / 0.474 |
| 24 h | val | 0.547 | 0.555 | −0.247 | 7.70 → 1.08 | 2.26 | 0.973 → 0.285 | 0.002 / 0.500 |
| 24 h | test | 0.544 | 0.546 | −0.224 | 7.76 → 1.03 | 2.53 | 0.946 → 0.267 | 0.002 / 0.452 |
| 48 h | val | 0.504 | 0.506 | −0.202 | 7.08 → 1.62 | 2.25 | 0.948 → 0.325 | 0.058 / 0.526 |
| 48 h | test | 0.508 | 0.504 | −0.211 | 7.30 → 1.71 | 2.55 | 0.941 → 0.302 | 0.033 / 0.489 |

**Physical-model comparison, re-run** (`results/comparisons/physical_baseline_gefsv12/`):
- Old-detector summaries: `physical_baseline_old_detector/`.
- Setup: 00Z starts, band 0.0375–0.485 Hz, physical labels in both runs.

| Lead | Detector | GEFSv12 | shape_v13 | ridge AR | Persistence | Pred. peaks v13 / persistence (true) |
|---|---|---|---|---|---|---|
| 12 h | old | 0.152 | **0.553** | 0.356 | 0.380 | 7.63 / 4.10 (4.29) |
| 12 h | new | **0.188** | −0.116 | 0.139 | 0.167 | 1.23 / 2.60 (2.51) |
| 24 h | old | 0.160 | **0.535** | 0.218 | 0.278 | 7.81 / 4.08 (4.11) |
| 24 h | new | **0.131** | −0.230 | −0.024 | 0.023 | 1.04 / 2.60 (2.59) |
| 48 h | old | 0.150 | **0.482** | −0.011 | 0.196 | 7.25 / 4.07 (4.12) |
| 48 h | new | **0.140** | −0.235 | −0.154 | −0.055 | 1.65 / 2.61 (2.61) |

The values are `peak_fidelity_SS`, and bold marks the best model per row.

Reading:
- The detector fix dominates. Physical labels alone move PF by at most 0.08.
- The old recall of about 0.95 was an artefact of the old detector. v13 forecasts carried 7–8
  "significant" peaks, mostly ripples, so nearly every true peak had one within 2 bins.
- With the new detector, v13's forecasts are close to unimodal: 1.0–1.7 peaks against 2.3–2.5 in
  the truth. Wind-sea partitions are almost never recalled at 24–48 h. This is the over-smoothing
  failure mode the peak panel was built to expose.
- The physical-model ranking reverses. v13 goes from best to worst on PF, below persistence, the
  ridge AR and GEFS.
- Persistence predicts 2.60 peaks against 2.51 true. So the new detector is not too strict for
  real spectra; the low count is v13's.

Caveats:
- One buoy (32012) and one experiment.
- v13 was trained with the old peak-loss windows and selected by the old PF, so these numbers
  score a model optimised for a different target. Whether a model retrained with the new windows
  and selected on the new PF also over-smooths is open.
- The combining order is our choice. Another order could change counts slightly.

## Decision

- **Kept.** The detector now matches the criteria as published and as the manuscript already
  describes them, and `evaluate()` labels on the physical spectrum.
- **Superseded numbers:**
  - every peak-panel number from `evaluate()` for the `shape` target before this commit;
  - [[026]]'s test panel;
  - [[028]]'s in-window rates;
  - v13's PF-based trial selection ([[009]]);
  - the pre-fix physical-baseline outputs.
- **Still open:** retraining shape_v13 (search and/or the [[026]] ablation) with the new windows
  and PF. Nothing in `results/` has been re-run with the new training windows.

## Related

- Code: `utils/spectral_partitioning.py` (`_combined_partitions`); `utils/spectral_peaks.py`;
  `nn/evaluate.py`; `nn/optimization.py::_prepare_dataloaders`; `nn/dataset.py`;
  `nn/training_loop.py::_peak_windows_for_batch`; `scripts/compare_physical_baseline.py`.
- Other decisions: [[008]] (criteria), [[009]] (PF objective), [[025]] (per-batch windows),
  [[026]] (loss ablation), [[028]] (window displacement check), [[029]] (physical labels).
- Manuscript: `02_methods.tex` §"Peak-resolved, partition-conditioned diagnostics". Its
  criterion 3 wording was already right; it needs one clause on combining. Any peak-panel
  number, and the GEFS comparison, must be taken from post-fix runs only.
