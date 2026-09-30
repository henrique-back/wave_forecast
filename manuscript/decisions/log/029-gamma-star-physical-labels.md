---
status: kept
date: 2026-09-30
commits: []
category: evaluation
---

# 029 — Wind-sea/swell labels (γ\*) must be computed on the physical spectrum, not the shape

## Context

While building the physical-model baseline (`scripts/compare_physical_baseline.py`), a
code check found that `nn/evaluate.py` passes the **unit-area shape** E(f)/m₀ to
`utils/spectral_peaks.py::peak_modality_metrics` for the `shape` target. That function
labels each true partition with γ\* = S(f_p)/S_PM(f_p) (`utils/spectral_partitioning.py::
classify_partition`). S_PM is the Pierson–Moskowitz reference in m²/Hz, so γ\* is only
meaningful for a physical spectrum. On a shape, γ\* is inflated by 1/m₀: at Hs = 2 m,
m₀ = 0.25 m², so γ\* reads four times too high. `scripts/check_peak_windows.py` already
labelled on the physical density for this reason (its docstring says so), but nothing
recorded the consequence for `evaluate()`.

## Change

New code only; `nn/evaluate.py` is **unchanged**.
- `nn/spectrum_eval.py::compute_shape_final_metrics(..., m0_true=...)` passes
  `(shape · m₀_true, true_shape · m₀_true)` to the peak panel. The labels therefore use
  physical units. Every other peak metric is invariant to that common scaling.
- With `m0_true=None`, it reproduces `evaluate()`'s current behaviour; the parity test in
  `tests/test_shape_final_metrics.py` pins this.
- The physical-model comparison uses the physical labels.

## Evidence

**Buoy 32012 test split**, all 2,632 raw hourly spectra (full 47-bin grid, measured
2026-09-30):
- Peak geometry is identical for shape and physical input in all 2,632 spectra. Only the
  labels move.
- 2,211 of 11,088 partitions (19.9%) change label: 2,210 from wind sea (shape space) to
  swell (physical), and 1 the other way.

**Illustration on the 64-bin test grid of `tests/test_spectral.py`:** a single JONSWAP peak
with Tp ≤ 12 s is labelled wind sea in shape space for every Hs from 0.5 to 2 m. Its
physical γ\* at Tp = 10 s is 0.03–0.55 (swell). The shape-space labels are biased
systematically towards wind sea, not scattered at random.

## Decision

- **Kept** for the new scorer and the physical-model comparison.
- **Still open** for `nn/evaluate.py`. Fixing it there needs the loader to carry m₀ (for
  the `shape` target, `prepare_y` discards it). It would also change historical numbers,
  so it is a separate, deliberate change.

**Affected by the old labels until then:**
- every `_windsea`/`_swell` peak metric (and `Peak_windsea_n`/`Peak_swell_n`) that
  `evaluate()` reports for the `shape` target;
- the loss-ablation test panel ([[026]]);
- `peak_fidelity_SS`, the wind-sea/swell-averaged PF criterion used to **select
  shape_v13** (`nn/optimization.py::_compute_val_score`) ([[009]]).

Pooled peak metrics, Tm02, and the partition-window geometry (and hence `SoftPeakHeightLoss`,
which uses windows and not labels) are unaffected.

**Update 2026-09-30 ([[030]]):** the `evaluate()` item is closed. `_prepare_dataloaders`
attaches the true m₀ to the val/test datasets for the `shape` target, and `evaluate()` scales
both final-step shapes by it before the peak panel. The same commit also changed the peak
detector itself ([[030]]), so post-fix wind-sea/swell numbers differ from the ones above for
both reasons; [[030]] separates the two effects.

## Related

- Code: `nn/evaluate.py` (shape block, `peak_modality_metrics` call);
  `nn/spectrum_eval.py::compute_shape_final_metrics`; `utils/spectral_partitioning.py`;
  `scripts/check_peak_windows.py`; `scripts/compare_physical_baseline.py`.
- Other decisions: [[008]] (peak detection and γ\* criterion), [[009]] (PF objective),
  [[026]], [[028]].
- Manuscript: `02_methods.tex` §"Peak-resolved, partition-conditioned diagnostics". The
  method it describes (labels from the true spectrum) is correct; the implementation
  behind the reported wind-sea/swell numbers is not yet. No reported number should be
  quoted until the evaluate() fix is decided (manuscript `CLAUDE.md` §0.9).
