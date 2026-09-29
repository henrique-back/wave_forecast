---
key: filoche2026postprocessing
title: Site-Specific Post-Processing of Spectral Wave Forecast by Learning From Buoy Measurements
authors: Filoche, A.; Hansen, J.; Vinsen, K.; Dawson, T.
year: 2026
venue: "JGR: Machine Learning and Computation, 3(5), e2026JH001454"
relevance: high
source_file: literature/filoche2026postprocessing.pdf — published version (20 pp., complete, open access CC BY-NC 4.0), supplied by the author 2026-09-29 and read in full. First found the same day via find-reference websearch and filed at abstract level (metadata via Crossref), with numbers taken from the authors' Zenodo code snapshot (doi:10.5281/zenodo.22557273). The snapshot numbers have now been checked against the paper, and the paper's values are used below.
---

# Site-Specific Post-Processing of Spectral Wave Forecast by Learning From Buoy Measurements

**Filoche, A., Hansen, J., Vinsen, K., Dawson, T. (2026).** *Journal of Geophysical Research:
Machine Learning and Computation*, 3(5), e2026JH001454. DOI: 10.1029/2026JH001454. Received
28 May 2026, accepted 2 September 2026, published online 17 September 2026. The authors are at the
University of Western Australia (Oceans Institute; ICRAR).

## Summary
A deep-learning post-processor for the ECMWF operational wave forecast at one deep-water site in
the Browse Basin, off north-west Australia.
- **Forecast input.** The operational ecWAM directional spectrum from MARS, at the nearest native
  grid node (13.75°S, 123.26°E), taken without horizontal interpolation. It is issued twice daily,
  out to 5 days at 3 h steps (40 lead times), with 36 directions × 36 frequencies. Only the first
  29 frequencies, approximately 0.034–0.49 Hz, are kept (§2.1).
- **Target.** The 1D spectrum E(f) from a Datawell Waverider Mk4 moored in about 250 m of water.
  Its 100-bin spectrum is interpolated onto the 29-bin ECMWF grid, then rescaled so that m₀
  (and hence Hs) is preserved (§2.2, Eq. 1). The buoy data belong to Shell Australia and are not
  public.
- **Architecture.** A 3D-CNN encoder-decoder with convolutional block attention modules, adapted
  from Archambault et al. (2024). Context variables (tidal water level, and six date and lead-time
  encodings) modulate it through FiLM (§3.2). It is **not** a transformer; the abstract's
  "attention-based encoder-decoder" refers to these attention modules. A cross-attention option
  was offered to the search and not selected.
- **Training and selection.**
  - The selected configuration trains on an MAE loss computed on log-standardised spectra (Eqs.
    3–5).
  - A TPE search (500 configurations) selects on validation Hs RMSE, averaged over all lead times.
  - The final prediction is the mean of an ensemble that differs only by random initialisation.
- **Split (§3.3.1).** Chronological: training January 2020–June 2022 (1,520 samples), validation
  July–December 2022 (368 samples), **test January–October 2023**. Ten-day buffers separate the
  splits, because the history context is 5 days long.
- **Verification (§4.2).** Bulk parameters Hs, Tp and Tm02 are computed from the ECMWF 2D spectrum
  and from the predicted 1D spectrum. They are also computed separately for swell and wind sea,
  split by a **hard 0.1 Hz frequency cut** chosen from the site climatology. The Table 2 headers
  print the bands as [0.04, 0.1] and [0.1, 0.5] Hz. No persistence baseline is reported.

**Table 2 (p. 11), test-set RMSE.** Hs is in m, and Tp and Tm02 are in s. The paper also has a
single-model "ML" row, omitted here.

| Lead | Model | Total Hs | Total Tp | Total Tm02 | Swell Hs | Swell Tp | Swell Tm02 | Wind-sea Hs | Wind-sea Tp | Wind-sea Tm02 |
|---|---|---|---|---|---|---|---|---|---|---|
| +3 h | ECMWF | 0.16 | 3.38 | 0.66 | 0.15 | 1.89 | 0.74 | 0.14 | 1.41 | 0.45 |
| +3 h | ML-Ensemble | 0.17 | 2.71 | 0.60 | 0.12 | 1.37 | 0.54 | 0.15 | 1.20 | 0.40 |
| +24 h | ECMWF | 0.16 | 3.53 | 0.70 | 0.14 | 2.05 | 0.75 | 0.14 | 1.40 | 0.47 |
| +24 h | ML-Ensemble | 0.17 | 2.95 | 0.64 | 0.11 | 1.32 | 0.52 | 0.16 | 1.27 | 0.42 |
| +60 h | ECMWF | 0.23 | 3.71 | 0.78 | 0.15 | 2.26 | 0.79 | 0.21 | 1.46 | 0.50 |
| +60 h | ML-Ensemble | 0.23 | 2.93 | 0.70 | 0.11 | 1.38 | 0.54 | 0.22 | 1.28 | 0.45 |
| +120 h | ECMWF | 0.25 | 3.79 | 0.92 | 0.16 | 2.22 | 0.81 | 0.24 | 1.47 | 0.59 |
| +120 h | ML-Ensemble | 0.25 | 2.95 | 0.88 | 0.12 | 1.39 | 0.57 | 0.25 | 1.33 | 0.55 |

**Other findings:**
- *Swell improves; wind sea does not.* Swell Hs RMSE falls by about 20–27% and swell Tp by up to
  39%. Wind-sea Hs is "consistently degraded", so **total Hs never beats ECMWF** (§4.2.1).
- *Shape parameters improve.* Total Tm02 and total Tp improve at every lead.
- *ECMWF over-predicts the 0.08–0.1 Hz band* at 120 h, while the correction slightly
  under-predicts swell energy (§4.2.3, Fig. 10).
- *The swell-arrival correction is real.* ECMWF's premature swell arrival appears as a
  phase-shifted pair of error principal components, and the correction reduces it (§4.2.4).
  - In the case study (Fig. 14), the corrected forecast places the arrival more than 60 h after
    ECMWF's.
- *The ML adds a negative bias above 0.1 Hz,* under-predicting wind-sea energy (§4.2.2).
- *The authors name over-smoothing as a cause:* "this ensembling may also smooth narrow or highly
  variable wind-sea peaks. Combined with the relative weighting induced by the logarithmic MAE,
  this may contribute to the conservative wind-sea estimates" (§4.2.1).
- *Buoy history hurt.* Five days of buoy history (40 spectra at 3 h) "surprisingly degrades
  performance" and "was explicitly rejected during hyperparameter optimization" (§4.1, §5.2.3).
  - The authors had expected it to help, because the operational ECMWF forecast assimilates
    satellite altimeter Hs but not buoy observations.
  - They attribute the failure to their architecture: it "failed to condition the correction on
    recent observations".
- *The 2D input beat the 1D input,* so directional information carries skill (§4.1).
- *The ensemble is under-dispersive,* so its spread is not usable as uncertainty (§4.3).

**Checked against the Zenodo snapshot** that the first filing relied on:
- The ECMWF values are identical.
- The ML-Ensemble values differ. For example, at +120 h the paper gives swell Hs 0.12 and swell
  Tp 1.39, where the snapshot gave 0.13 and 1.38.
- The snapshot's "test = calendar 2023" is wrong: the paper tests January–October 2023.

## Relevance to this manuscript
High, for four reasons.

*A published physical-model spectral baseline.* This is one of very few published verifications
of an operational numerical model's frequency-spectrum forecast against a buoy by lead time. It is
also the only one found in the 2026-09-29 search that reports buoy-verified swell and wind-sea
errors separately at each lead (`rogers2025espc` splits by band too, but verifies against
satellite spectra). The ECMWF Tm02 rows do not change when a spectrum is rescaled, so they stay
comparable if the manuscript reports only the shape forecast. They are context, not a
head-to-head, because four things differ:
- site: tropical north-west Australia, vs the southeast Pacific;
- period: 2023, vs 2017;
- frequency grid: 29 bins over 0.034–0.49 Hz;
- partition method: a hard 0.1 Hz cut, vs trough-to-trough partitions labelled by γ\*.

*It narrows the Introduction's layer-2 gap claim.* It is a learned forecast of the full 1D
spectrum at a buoy over multi-day leads, so "none forecast the full spectrum" (`../../CLAUDE.md`
§0.3) does not hold without qualification. The distinction that survives is the input. It
post-processes a numerical forecast, with the ECMWF spectrum as its main input and the forecast
issuance as its time reference. This manuscript forecasts from the buoy's own history, with no
numerical-model input.

*Buoy history degraded their model.* A reviewer may put this to the premise that a buoy's own
history carries forecast skill. It does not contradict it: in their setup the physical forecast is
already present, and the authors attribute the failure to their own architecture, not to the
information content. It does frame the comparison. A buoy-history model is most useful where a
numerical forecast is unavailable, too costly, or poorly suited to the site.

*Direct support for the manuscript's measurement argument.*
- Their model is selected on an aggregate bulk score (validation Hs RMSE, averaged over leads).
- They name ensemble averaging and a log-MAE loss as likely causes of smoothed, under-predicted
  wind-sea peaks.
- Their correction improves the parameters that depend on spectral shape (Tp, Tm02, swell) but
  never total Hs. It is a small, independent instance of learned models gaining on shape rather
  than on magnitude, which fits a shape-focused framing.

## Suggested use
- **Discussion, physical-baseline paragraph.** Cite for the ECMWF RMSE of Tm02 and partition Hs by
  lead time (Table 2) as published context, alongside the planned same-site numerical-model
  comparison. State the site, period, grid and partition-method caveats in the same sentence.
- **Introduction, the "work that does engage with the spectrum" paragraph.** Cite as a third line
  of prior work, next to the estimation and spatial-forcing lines already cited: learned
  correction of a numerical model's spectral forecast at a station. The paragraph's closing
  sentence ("from a station's own recent spectral history has received considerably less
  attention") still holds with it added.
- **Discussion, over-smoothing.** Cite the §4.2.1 sentence on ensemble averaging and log-MAE
  smoothing wind-sea peaks, as a wave-domain acknowledgement of the pathology the loss and
  evaluation panel target. It pairs with `benbouallegue2024rise` for weather.
- **Discussion, buoy history as input.** If a reviewer raises their negative buoy-history result,
  answer with their own attribution (§5.2.3: the architecture "failed to condition the correction
  on recent observations") and the difference in setting (a numerical forecast was available).
  Do not claim that they showed buoy history carries no skill.
