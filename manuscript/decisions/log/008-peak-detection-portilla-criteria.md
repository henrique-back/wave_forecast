---
status: kept
date: 2026-08-06
commits: [b156e02]
category: evaluation
---

# 008 — Peak detector: scale-free `prominence_frac` heuristic → Portilla et al. (2009) 4-criterion test

## Context

Multimodal (wind-sea + swell) sea states need a peak detector for `peak_modality_metrics`
(`Peak_Count`, `Peak_Height_RelError`, `Peak_Separation_Recall`, wind-sea/swell-conditioned
breakdowns). The original detector used a `prominence_frac` heuristic — a peak accepted if
its height relative to each spectrum's own max exceeded a fixed fraction — with that
fraction tuned empirically by sweeping the constant against one buoy's test set. This is
scale-free by construction (relative to each spectrum's own max) but has no physical
grounding and was fit to a single buoy, raising generalization and defensibility concerns
for a published metric.

## Change

Replaced with `utils/spectral_partitioning.py::find_significant_peaks`, implementing the
Portilla et al. (2009, section 2b.2) four-criterion significant-peak test: a peak is
rejected if `fp > f_max` (default 0.4 Hz), its trough-to-trough partition energy is below
`energy_frac` (default 0.05) of total spectrum energy, it has fewer than `min_bins` (default
2) bins on either side before the next trough, or it is sandwiched between two higher-energy
neighboring peaks. The same module's `classify_partition`/`classify_partitions` additionally
labels each peak `'wind_sea'` vs. `'swell'` via `γ* = S_obs(fp)/S_PM(fp)` against the
Pierson-Moskowitz reference (threshold 1.0, per Violante-Carvalho 2009).

## Evidence

No quantitative before/after comparison exists in `results/` between the two peak detectors
directly (they were never run side-by-side on the same held-out set to produce a metrics
delta) — this was a methodological substitution, not a performance ablation. The
justification is the citation trail itself: replacing a single-buoy-tuned, dimensionless
heuristic with a published, physically-motivated, parameter-light test that generalizes
without per-buoy retuning.

## Decision

Kept. This is the peak detector `nn/evaluate.py` and `utils/spectral_peaks.py` currently
call for all peak-modality and Tm02-per-partition metrics (`peak_modality_metrics`,
`Tm02_RMSE_windsea`/`_swell`).

## Related

Code: `utils/spectral_partitioning.py::find_significant_peaks`,
`utils/spectral_peaks.py::find_spectral_peaks`
Other decisions: [[009]] (downstream use of this detector's output as an Optuna objective)
Manuscript section this might feed: Methods — evaluation metrics, specifically the
peak-detection/partitioning paragraph. Cite Portilla et al. (2009) and Violante-Carvalho
(2009) — confirm both are in `literature/refs.bib` before drafting (per manuscript `CLAUDE.md`
section 3, don't cite from memory).
