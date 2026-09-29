---
key: minuzzi2023lstm
title: A Deep Learning Approach to Predict Significant Wave Height Using Long Short-Term Memory
authors: Minuzzi, F.C.; Farina, L.
year: 2023
venue: Ocean Modelling, 181, 102151
relevance: high
source_file: literature/minuzzi2023lstm.pdf — published Elsevier version (18 pp., complete), supplied by the author 2026-09-29 and read in full. First found via find-reference websearch 2026-09-21 and read then only via arXiv:2201.00356 (preprint)
---

# A Deep Learning Approach to Predict Significant Wave Height Using Long Short-Term Memory

**Minuzzi, F.C., Farina, L. (2023).** *Ocean Modelling*, 181, 102151.
DOI: 10.1016/j.ocemod.2022.102151. Received 15 October 2021, accepted 22 November 2022, online
2 December 2022. The metadata in `refs.bib` matches the published PDF. The authors are at the
Institute of Mathematics and Statistics, UFRGS (this manuscript's own institute), and Farina is
also at CECO/UFRGS. Funded by ONR Global and CAPES.

## Summary
A univariate LSTM forecasts Hs at the locations of seven Brazilian Navy PNBOIA buoys, from Rio
Grande (31°S) to Fortaleza (3°S), at 6, 12, 18 and 24 h lead times. The network has three
layers (64/48/32 units) and is trained with an MAE loss and Adam in TensorFlow/Keras. Each
buoy's verification period is one month (about 744 hourly steps). Code is at
github.com/felipeminuzzi/lstm-ocean.

**The protocol differs from ours in three ways that matter for comparison:**
- *Most experiments are trained and scored on ERA5 reanalysis, not observations.* ERA5 Hs at
  each buoy location is both the training data and the reference in the metrics. Only buoys 1
  and 2 are also trained and scored on buoy data, which has been available since April 2009.
  Outliers were removed with k-nearest neighbours: 1.6% of buoy 1's data and 14.5% of buoy 2's.
- *The model is retrained for every verification time.* The training set runs up to that time
  minus the lead time, and the model makes a single one-hour prediction across the gap (their
  Fig. 4). The training start date is tuned per lead time against MAPE (Table 2, e.g. from 2003
  for 6 h and from 1987 for 24 h). This is walk-forward retraining, not a fixed train/test
  split.
- *"Accuracy" means 100% − MAPE* (footnote to §5.1). No persistence or climatology baseline is
  reported.

**Results.** Against ERA5, accuracy is highest at 6 h in every location. It reaches 97.25% at
buoy 7 and stays above 85% at all lead times for buoy 2. MAPE rises from 2-6% at 6 h to 6-24%
at 24 h, depending on location (Table 4). Trained and scored on buoy data, accuracy is 87.04%
(6 h) falling to 73.83% (24 h) at buoy 1, and 81.74% falling to 70.43% at buoy 2. At 6 h this
beats ERA5's own agreement with the buoys (80.67% and 80.59%). From that, the authors *infer*
that the LSTM "can improve the 6 h forecast which uses a physical model". **No physical-model
forecast is actually run or compared.**

Adding inputs did not help. Neither Tp and U10 nor the four ERA5 variables best correlated with
Hs (mean swell and wind-sea periods) improved on univariate Hs. Using Tp and U10 *without* Hs
lowered accuracy, though it stayed above 70%. Training takes about 14 min per lead time, and a
single prediction takes 2.62 × 10⁻⁶ s (Xeon, 20 cores, RTX 2080 Ti). The authors assert, without
measuring it, "a large decrease in computational time if compared to traditional physical
models".

**Smoothing toward the mean.** The paper reports this failure mode repeatedly:
- At buoys 5 and 7 the variability "leads the LSTM (and also the reanalysis) to smooth the
  prediction around a mean value".
- With short training sets, the 18 and 24 h predictions "will fail to predict peaks or valleys
  since the network will approach the average value of the period".
- In the Tp/U10-only run, "the peaks are not reproduced by the LSTM" at 24 h.
- The review of related work also notes that earlier ANN wave forecasts (Londhe and Panchang
  2006) had "considerable under-predictions in the highest peaks".

**Internal inconsistency to be aware of:** the text calls buoys 5 and 7 "shallow water" and
buoy 2 "deeper water". Table 1, however, lists a depth of 200 m for every buoy except buoy 3
(2164 m). Do not cite depth-dependent conclusions from this paper.

## Relevance to this manuscript
High. It is a peer-reviewed, same-institute precedent for data-driven, buoy-location Hs
forecasting. It serves three purposes:
- **Surrogate framing, in the Introduction (currently cited at `01_introduction.tex:56`).** It
  poses the question verbatim: "can data-driven models, with help of the artificial
  intelligence, act as a physical model surrogate, with computational time and accuracy that are
  superior to the latter?". It names WAVEWATCH III and SWAN as the physics-based
  state of the art. The abstract concludes that the methodology "can be used as a surrogate to
  the computational expensive physical models". It credits the question itself to Boukabara et
  al. (2019, 2022). The manuscript's "prompting the question of whether such models can act as
  physical-model surrogates at buoy scale" is therefore fair. Cite it for *posing* the question
  and for its computational-cost claim. Do **not** cite it as evidence that an LSTM beats a
  numerical model: that is an inference from ERA5, not a head-to-head comparison.
- **The honest trade-off (manuscript `CLAUDE.md` §0.6).** Its conclusion states the costs
  alongside the speed advantage. Networks are "partially 'black box' models", which makes
  "the results difficult to analyse from a physical point of view". Their processing time
  "scale[s] with data size and regions", and a global prediction "might be not feasible". Both
  points directly support the "speed at the expense of interpretability" clause in the
  Introduction.
- **Scalar-only lineage and over-smoothing.** It forecasts Hs only, and extends the
  bulk-parameter-only lineage from `zhou2021emdlstm` (2021) towards `kim2026metoformer` (2026).
  Its repeated smoothing-toward-the-mean observations are a wave-domain, recurrent-model
  instance of the over-smoothing failure mode that `benbouallegue2024rise` documents for weather.
  That is small, but useful, support for the Introduction's layer-4 argument.

It does **not** support an "unavailable at remote or local sites" framing. The nearest it comes
is noting that ERA5 is "usually not available on a daily basis", so only the buoy-trained
variant "could be adapted to be used as a possible operational forecast".

## Suggested use
- Introduction: keep it on the surrogate-question sentence, and optionally pair it with the
  interpretability and scaling caveats from its conclusion.
- Discussion: use it as a scalar-Hs, buoy-location comparator, citing its protocol rather than
  its numbers. Do not set its "accuracy" figures beside our skill scores. They are 100% −
  MAPE, mostly scored against ERA5 rather than observations, have no persistence baseline, and
  come from per-step walk-forward retraining, so they are not comparable.
- Its "extra inputs did not help" result sits comfortably beside `jiang2024comment`'s critique
  of complex models for scalar Hs. If cited, cite it as a finding for scalar Hs, not as a claim
  about spectra.
