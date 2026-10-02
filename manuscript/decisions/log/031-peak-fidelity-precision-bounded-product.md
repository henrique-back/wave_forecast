---
status: kept
date: 2026-09-30
commits: []
category: objective-metric
---

# 031 — `peak_fidelity_SS` → `peak_fidelity`: a false-positive term, and two bounded factors

## Context

[[030]] fixed the peak detector and, in doing so, showed that shape_v13's selection criterion had
been reading a broken panel. Re-scoring moved v13's PF from +0.560 to −0.095 (12 h test). That
entry left the metric itself alone, but two faults in it are independent of the detector and
survive the fix.

**No false-positive term.** `peak_fidelity_SS` was
`mean(Peak_Separation_Recall_windsea, _swell) − mean(Peak_Height_RelError_windsea, _swell)`.
Recall counts only true peaks the model found; nothing in the repo ever asked whether a *predicted*
peak corresponds to anything (a repo-wide search found no precision, no F1, no matching — the
predicted peak set was used only for `Peak_Count_Pred_Mean` and the nearest-neighbour recall test).
A model is therefore never charged for inventing peaks, and the criterion is maximised by
over-segmenting.

That is not hypothetical. Under the old detector v13 emitted 7.89 peaks against 4.22 true and
scored 0.560 on recall of 0.949; measured precision was 0.484 (one-to-one matched), i.e. about half
its peaks corresponded to nothing. PF could not see it. The immediate cause of the inflated counts
was the detector, not the model gaming the metric — but a criterion that cannot distinguish "found
the peaks" from "emitted peaks everywhere" gave no signal that anything was wrong, at any point
across three lead times and 80 trials each.

**Unbounded minus bounded.** `recall ∈ [0,1]` but `rel_err ∈ [0,∞)`, subtracted at an implicit 1:1
weight that was never justified. One badly-missed peak height can dominate a term that is
structurally a fraction, and the score has no interpretable range.

Separately, `nn/optimization.py::objective` never wrote any peak metric to `trial.user_attrs`, so
the only trace a finished trial kept of PF was `trial.value` — the old number under the old
definition. That is why v13's 80 trials per lead could not be re-ranked when the detector was fixed
and had to be discarded instead.

## Change

`utils/spectral_peaks.py::peak_modality_metrics`:
- New `Peak_Separation_Precision`: the mirror of the existing recall test walked from the predicted
  side — fraction of predicted peaks with a true peak within `bin_tolerance`. Pooled (micro) over
  all samples, NaN when nothing was predicted anywhere.
- **Not** split by `_windsea`/`_swell`. Labels come from `classify_partition` on the true
  partition; a false positive has none to inherit. Labelling it from the prediction would break
  this module's "geometry and labels from the TRUE spectrum only" convention, which
  `SoftPeakHeightLoss` shares.
- A plain membership test, not one-to-one matching. Measured under the [[030]] detector, matched
  and unmatched precision agree to three decimals almost everywhere (0.681/0.681, 0.797/0.797,
  0.800/0.799; largest gap 0.466→0.448): with ~2.5 peaks per spectrum and a median 16-bin window
  there is no room to claim one true peak with a cluster of predicted ones. Under the *old*
  detector the same pair differed by 0.689 vs 0.484, so the earlier case for matching was itself a
  detector artefact.

`nn/optimization.py::_compute_val_score` — renamed `peak_fidelity_SS` → `peak_fidelity` and
redefined:

```
R  = nanmean(Peak_Separation_Recall_windsea, _swell)       macro, [0, 1]
P  = Peak_Separation_Precision                             pooled, [0, 1]
F1 = 2PR / (P + R)                    (0 when P + R == 0)
H  = 1 - min(nanmean(Peak_Height_RelError_windsea, _swell), 1)
peak_fidelity = F1 * H                                             [0, 1]
```

Recall stays macro so a model cannot ignore the rarer regime. A NaN precision (nothing predicted)
is read as 0.0, not as missing data. `float('-inf')` on both-label-NaN recall/rel_err and the
hard `KeyError` on absent keys are unchanged.

The product rather than a weighted sum: both terms are now bounded, so a sum would still let a
model trade one away (a forecast finding no peaks at all would still score ~0.3 on height alone).
Under the product that is 0.

The rename is deliberate. It is not a skill score — no baseline appears in it — and the `_SS`
suffix invited exactly the reading that a positive number meant "better than the baseline", which
v13 was not. It also makes old and new numbers un-confusable: `_compute_val_score` raises
`ValueError` on unknown names, so any caller left on `peak_fidelity_SS` fails loudly instead of
silently receiving a differently-scaled number. `results/comparisons/peak_detector_030/rescore_shape_v13.py`
is deliberately left on the old name and marked frozen — its committed JSON is [[030]]'s evidence
under the old definition, and re-scoring it under this one would contradict the entry it supports.

Also:
- `nn/optimization.py::objective` now writes the peak panel (recall/precision/rel_err per label,
  the count pair, `Peak_*_n`, `Tm02_RMSE/Bias_*`) to `trial.user_attrs`, so a future revision can
  re-rank a finished study offline instead of discarding it.
- `scripts/ablate_loss.py` does the same, additionally closing the `Tm02_Bias_windsea/_swell` gap
  that `scripts/evaluate_ablation_phases.py` existed to work around.
- `scripts/compare_ablation_phases.py` prints a pooled precision + `#pred` vs `#true` block. That
  count pair is the cheapest guardrail available: it read 7.89-vs-4.22 before the detector fix and
  1.03-vs-2.53 after, and it catches both failure directions regardless of the score's formula.
- Fixed a latent ordering bug in `scripts/ablate_loss.py::_fixed_weights_for_phase`: the
  `_read_prior_weight("kl", ...)` disk read ran before the unknown-phase check, so a typo'd phase
  name raised `FileNotFoundError("run --phase kl first")` — wrong error, misleading advice. It only
  looked correct while a completed `kl` phase happened to exist for the current `STUDY_VERSION`.

Tests: `tests/test_spectral_peaks.py::TestPeakSeparationPrecision` (a prediction containing the
true peak plus one invented peak — recall stays 1.0, precision drops to 0.5, the exact case the old
metric was blind to; identical-input and no-predicted-peaks cases);
`tests/test_optimization.py::TestPeakFidelity` rewritten, adding a spurious-peak penalty test, a
bounded-score test at `rel_err = 50`, a NaN-precision test and a test that the old name is
rejected; `tests/test_shape_final_metrics.py` gains precision in the `evaluate()`/scorer parity and
m₀-scale-invariance lists (precision is pure geometry, so it must not move under the [[029]]
rescaling).

## Evidence

The four models of the physical-model comparison, scored under the new metric on the same 00Z
issue times and 45-bin band (`results/comparisons/physical_baseline_gefsv12/lead_{12,24,48}h/arrays.npz`):

| Lead | Model | #true | #pred | recall | precision | rel. err. | `peak_fidelity` |
|---|---|---|---|---|---|---|---|
| 12 h | GEFSv12 | 2.51 | 1.94 | 0.625 | 0.808 | 0.408 | **0.408** |
| 12 h | ridge AR | 2.51 | 1.84 | 0.572 | 0.787 | 0.375 | 0.387 |
| 12 h | persistence | 2.51 | 2.60 | 0.621 | 0.604 | 0.431 | 0.342 |
| 12 h | shape_v13 | 2.51 | 1.23 | 0.335 | 0.682 | 0.419 | 0.244 |
| 24 h | GEFSv12 | 2.59 | 1.96 | 0.578 | 0.764 | 0.389 | **0.378** |
| 24 h | ridge AR | 2.59 | 1.64 | 0.436 | 0.684 | 0.403 | 0.291 |
| 24 h | persistence | 2.59 | 2.60 | 0.553 | 0.554 | 0.494 | 0.271 |
| 24 h | shape_v13 | 2.59 | 1.04 | 0.273 | 0.664 | 0.462 | 0.182 |
| 48 h | GEFSv12 | 2.61 | 1.96 | 0.580 | 0.767 | 0.383 | **0.383** |
| 48 h | persistence | 2.61 | 2.61 | 0.504 | 0.504 | 0.528 | 0.231 |
| 48 h | ridge AR | 2.61 | 1.30 | 0.339 | 0.684 | 0.445 | 0.226 |
| 48 h | shape_v13 | 2.61 | 1.65 | 0.288 | 0.457 | 0.485 | 0.165 |

Every score is inside [0,1]. The ordering matches [[030]]'s independent re-scoring: GEFSv12 best at
every lead, shape_v13 last at every lead, below persistence. Precision separates the two failure
directions that recall alone conflates — persistence has balanced P≈R with a near-exact peak count,
while shape_v13 under-predicts peaks (1.0–1.7 against 2.5–2.6) and so scores poorly on recall
despite respectable precision.

Supporting measurement of matched vs unmatched precision under both detectors:
`results/comparisons/shape_v13_vs_shape_v12_vs_linear_baseline_peak_fidelity/precision_f1.json`
(old detector) and `precision_f1_newdetector.json` (new).

## Decision

- **Kept.** The criterion now has a false-positive term and a bounded range.
- **Supersedes [[009]]** (`exploratory — open`), which introduced `peak_fidelity_SS` and flagged as
  unresolved whether the whole-spectrum regression it accepted was worth the cost. That question is
  now moot in its original form: the recall it traded RMSE for was not measuring what it appeared to.
- **Superseded numbers:** every `peak_fidelity_SS` value anywhere, on top of those [[030]] already
  listed. The two are not on the same scale and must never be compared.
- **Still open:** the loss ablation re-run under this metric and the [[030]] training windows
  (`STUDY_VERSION = lossablation_v3`), and the v14 production search that depends on its outcome.
  v14 also un-pins `FIXED_HEAD_DIM`/`FIXED_NHEAD`: [[027]] pinned them on a tally over studies that
  were themselves selected under the superseded criterion, so that evidence no longer stands.

## Related

- Code: `utils/spectral_peaks.py`; `nn/optimization.py` (`_compute_val_score`, `objective`);
  `nn/spectrum_eval.py`; `scripts/ablate_loss.py`; `scripts/compare_ablation_phases.py`;
  `scripts/compare_physical_baseline.py`; `scripts/optimize.py`; `scripts/train.py`.
- Other decisions: [[008]] (detection criteria), [[009]] (the metric this replaces), [[026]] (the
  ablation to re-run), [[027]] (the head_dim/nhead pinning this undermines), [[029]] (physical
  labels), [[030]] (the detector fix that exposed all of it).
- Manuscript: `02_methods.tex` § "Core metrics and model selection" (boxed equation) and
  § "Peak-resolved, partition-conditioned diagnostics" (precision added to the metric list). No PF
  number may be quoted from a pre-031 run.
