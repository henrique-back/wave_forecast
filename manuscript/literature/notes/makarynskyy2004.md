---
key: makarynskyy2004
title: Improving Wave Predictions with Artificial Neural Networks
authors: Makarynskyy, O.
year: 2004
venue: Ocean Engineering, 31(5-6), 709-724
relevance: high
source_file: literature/makarynskyy2004.pdf
---

# Improving Wave Predictions with Artificial Neural Networks

**Makarynskyy, O. (2004).** *Ocean Engineering*, 31(5-6), 709-724. DOI: 10.1016/j.oceaneng.2003.05.003.

**Correction 2026-09-29:** earlier versions of this note, and the Introduction draft built from
them, described the paper as ANNs correcting *numerical* wave forecasts. **That is wrong.** A
re-read of the PDF (Sections 1, 3 and 4) confirms that **no numerical wave model is run or
used**. The paper cites the shortcomings of numerical models only as motivation.
- The "initial forecasts" are produced by a neural network. Data set 1 is used "for training
  neural networks producing the initial forecasts".
- The "correction" is a second network applied to those network forecasts.
- `01_introduction.tex` has since been reworded ("ANNs forecasting wave height and period"), and
  `../../CLAUDE.md` §0.3 was corrected the same day.

## Summary
Feed-forward ANNs, three-layer and log-sigmoid, forecast hourly significant wave height and
zero-up-crossing period 1–24 h ahead at two Irish buoys. M1 is in the Atlantic (Mar 2001–Dec
2002) and M2 in the Irish Sea (May 2001–Dec 2002). The method has three stages, each a separate
network per parameter and site (Section 4):
1. **Initial forecast.** The buoy's own previous 48 h (48 input nodes) maps to the next 24 hourly
   values (24 output nodes).
2. **Correction.** A second network takes the 24 initial forecasts and outputs 24 corrected ones.
3. **Merging.** A third network takes the last 24 h of measurements together with the 24 initial
   forecasts (48 inputs) and outputs 24 values.

Each record is split into three equal consecutive parts:
- the first trains the initial-forecast networks;
- the second trains the correction and merging networks;
- the third is held out for validation.

Accuracy is reported as R, RMSE and SI = RMSE / observed mean, which is not bias-removed (Eqs.
1–3). Both correction approaches improve on the initial network forecasts.

## Relevance to this manuscript
High, and more so than first logged. It is an early precedent for forecasting from a **buoy's
own recent record**, the same input setting as this manuscript, but for scalar Hs/Tz only. So it
anchors the start of the scalar-only lineage (`../../CLAUDE.md` §0.3, layer 2) on the right
footing: an ANN *forecasting* bulk parameters, not *correcting a numerical model*.

## Suggested use
- **Introduction, the scalar-lineage sentence.** It is already cited there, now worded as ANNs
  forecasting wave height and period.
- **Do not cite it as ML post-processing of a numerical model.** For that line of work, cite
  `filoche2026postprocessing`.
- **Do not set its SI values beside ones normalised differently.** Its SI is RMSE over the
  observed mean, whereas `bidlot2002intercomparison` and `hanson2009pacific` use the bias-removed
  standard deviation over the mean.
