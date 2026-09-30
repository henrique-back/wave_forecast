---
status: kept
date: 2026-09-29
commits: []
category: training
---

# 028 — Peak-height loss windows stay fixed to the true spectrum (displacement check)

## Context

`SoftPeakHeightLoss` takes each predicted peak height as a soft maximum of the predicted
spectrum inside a trough-to-trough window of the **true** spectrum. A predicted peak with the
right height but shifted past the true trough would therefore be scored as missing. This
contradicted the Methods claim that the term "is insensitive to a peak's position and so
complements $W_2$", which came up while checking `02_methods.tex` against the code on
2026-09-29.

If this happened often, three changes were considered, all of which would mean rerunning the
loss ablation ([[026]]) and the v13 search:
- widening each window by the 2-bin recall tolerance;
- a soft, distance-weighted window instead of the hard trough-to-trough cut;
- comparing each true peak with its nearest predicted peak.

## Change

No change to the loss. The Methods text was corrected instead: the term ignores shifts within
a peak's own partition, and a peak displaced beyond its partition is penalised as missing, as
it also is by $W_2$. `scripts/check_peak_windows.py` was added to measure how often
displacement happens (uncommitted at the time of writing).

## Evidence

Source: `results/peak_window_check/shape_v13_lead_12h_val.txt`, produced by
`scripts/check_peak_windows.py`.

What was checked:
- **Checkpoint:** `results/shape_v13/shape/lead_12h/best_model.pt` (trial 119, smoothed
  validation score 0.5686). This is the local copy synced on 2026-09-18; the 12 h study kept
  running after that, so it is not necessarily the study's final best.
- **Data:** validation split, autoregressive inference (the evaluation path).
- **Partitions:** the first 4 windows per spectrum, the same cap the training loss applies.
- **Labels:** wind sea/swell computed on the physical density.

| Final step (n = 8,811 partitions) | All | Wind sea | Swell | Windows ≤ 5 bins |
|---|---|---|---|---|
| Nearest predicted peak inside the true window | 94.8% | 92.5% | 96.2% | 92.6% |

Distance from the true peak to the nearest predicted peak: 0 bins 34.9%, 1 bin 55.1%,
2 bins 7.4%, 3–5 bins 2.1%, more than 5 bins 0.5%.

Median relative height error:
- **Displaced partitions** (n = 462): 0.236 from the peak term, 0.422 against the nearest
  predicted peak.
- **Partitions with the peak in the window:** 0.235 from the peak term.

All 12 steps (n = 105,721, what the loss sees) give the same picture: 94.9% inside; median
errors 0.211 against 0.418 for displaced partitions and 0.213 in window.

Reading:
- Displacement is rare, at about 5% of partitions.
- In the displaced cases the peak term's error matches the in-window error, while the nearest
  predicted peak is further from the true height. So the peak outside the window is usually a
  different peak, not the true one shifted with its height intact. The term is not
  overcharging the case the Context worried about.

Caveats:
- One checkpoint, one split and one horizon (12 h).
- The model was trained with this term, so it may have learned to keep peaks inside the
  windows. A model trained without the term was not checked.
- Inference renormalises each fed-back step to unit area and training does not, so
  training-time predictions may differ slightly.
- Side finding, not decided here: predicted spectra have 7.88 significant peaks on average,
  against 3.69 in the true spectra. With about 8 predicted peaks per spectrum, a true peak
  often has a predicted one nearby by chance, which inflates the in-window rates above (and
  peak-separation recall) by the same effect.
- Only 34.9% of nearest predicted peaks sit on the true peak's own bin, against 55.1% one bin
  away. This was not investigated.

## Decision

Kept: the loss is unchanged and only the text was corrected. Re-measure if spurious predicted
peaks are reduced, since the in-window rates depend on how many peaks the model predicts, or if
a model trained without the peak term is ever compared.

**Update 2026-09-30 ([[030]]):** all numbers above use the old detector, which kept far too
many peaks (criterion 3 was effectively inactive and spurious partitions were not combined).
The 3.69 true and 7.88 predicted peaks per spectrum, and the 5-bin-scale windows, are symptoms
of that. Re-measure with the [[030]] detector before quoting them.

## Related

Code: `utils/loss.py::SoftPeakHeightLoss`, `nn/training_loop.py::_peak_windows_for_batch`,
`scripts/check_peak_windows.py`
Other decisions: [[008]] (peak detector), [[009]] (peak-fidelity objective), [[030]] (detector fix), [[025]]
(per-batch window detection), [[026]] (composite loss)
Manuscript section this might feed: Methods — loss function, peak-height term (already
updated).
