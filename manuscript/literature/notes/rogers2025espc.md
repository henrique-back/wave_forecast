---
key: rogers2025espc
title: Skill of Long-Range Forecasts of Ocean Wave Spectra from the Navy ESPC Version 2 System
authors: Rogers, W.E.; Janiga, M.A.
year: 2025
venue: NRL Memorandum Report; preprint arXiv:2510.06484
relevance: medium
source_file: no local PDF — full text of arXiv:2510.06484v2 (v1 7 Oct 2025, v2 9 Oct 2025; arXiv comments "NRL Memorandum Report, 44 pages") read 2026-09-29. First found via the gap-verification websearch 2026-09-18, when the note was written from search snippets; rewritten from the full text 2026-09-29.
---

# Skill of Long-Range Forecasts of Ocean Wave Spectra from the Navy ESPC Version 2 System

**Rogers, W.E., Janiga, M.A. (2025).** Naval Research Laboratory Memorandum Report.
Preprint: arXiv:2510.06484.

**Correction 2026-09-29:** the earlier, snippet-based note said the paper "stratifies
verification by wind-sea fraction, swell height, and wind-sea height — i.e. partition-conditioned
skill scoring". **That overstates it.**
- There is no stratification by wind-sea fraction. The model's total wind-sea fraction (TWSF) is
  used only to split energy into sea and swell heights.
- Swell and wind-sea heights are *verified quantities*, not strata.
- Those heights are verified only against the model's own analyses, never against observations.

## Summary
Skill assessment of the long-range forecasts from the wave component (WAVEWATCH III) of the
Navy's coupled ESPC v2 system. Only ensemble member 0 is evaluated. There are 26 long runs, one every 14 days from
6 September 2020 to 22 August 2021, evaluated out to 1080 h (45 days) (Sections 3.1 and 3.3). It evaluates seven
"wave height" parameters:
- energy in **four fixed frequency bands** (0.056–0.08, 0.08–0.11, 0.11–0.15 and 0.15–0.263 Hz);
- energy over all four bands combined;
- **swell height and wind-sea height**, taken from the model's bulk-parameter output (Executive
  Summary).

Sea and swell are split *inside the model*. A spectral component counts as wind sea when the
projected wind speed 1.7·U10·cos(θ − θw) exceeds its phase speed c(f), following the WW3 manual
(Section 3.1).

It uses two "ground truths":
1. **SWIM/CFOSAT satellite non-directional spectra**, bias-corrected band by band with the Ocean
   Station Papa buoy and a WW3 hindcast as intermediary. This is the only use of in-situ data.
   The four bands and the combined height are verified against it; sea and swell are not.
2. **The model's own analyses ("self-analysis").** All variables are verified against these,
   sea and swell included. The authors note it "will not reveal problems with model dynamics or
   calibration" (Section 3.3).

Conclusions (Section 4), all from self-analysis:
- the lowest-frequency band has the worst skill and the highest band the best;
- wind-sea height has worse skill than swell height;
- in week 2, the correlation for both is worse south of 20°S.

Per-band numbers (bias, RMSE, correlation, SI) appear only as figure insets (Figs. 13–17) and
were not transcribed here.

## Relevance to this manuscript
Medium, and narrower than first logged.
- It is a recent example of spectral verification by frequency band, plus separate sea and swell
  heights, for a numerical forecast.
- But it is **not** partition-conditioned verification against observations. The partitions are
  model-internal, and they are checked against the model's own analyses.
- `hanson2009pacific` remains the primary citation for partition-based verification against
  buoys.
- It gives no buoy-based comparator for this manuscript: its only in-situ data are Ocean Station
  Papa, used for calibration.

## Suggested use
- **Keep it as corroboration, not as a second example of Hanson-style verification.** It is cited
  at `01_introduction.tex:93`, where the text says Hanson's partition verification is "an
  approach more recently applied to long-range spectral forecasts". That is defensible only
  loosely: sea and swell heights are verified separately, but against self-analysis, and alongside
  fixed frequency bands rather than spectral partitions. If the sentence stays, consider
  qualifying it.
- **Do not cite it for partition-conditioned verification against observations,** nor for
  stratification by wind-sea fraction.
