---
key: filoche2026postprocessing
title: Site-Specific Post-Processing of Spectral Wave Forecast by Learning From Buoy Measurements
authors: Filoche, A.; Hansen, J.; Vinsen, K.; Dawson, T.
year: 2026
venue: "JGR: Machine Learning and Computation, 3(5), e2026JH001454"
relevance: high
source_file: web search, no local PDF — found 2026-09-29 during the search for physical-model spectral verification to compare against. Metadata and abstract read from Crossref (api.crossref.org) on 2026-09-29. The lead-time error table below comes from the authors' reviewer-facing code snapshot on Zenodo (doi:10.5281/zenodo.22557273, published 2026-09-07), read the same day. The article is open access (CC BY-NC 4.0), but the Wiley PDF blocks automated download (HTTP 403), so the full text has NOT been read.
---

# Site-Specific Post-Processing of Spectral Wave Forecast by Learning From Buoy Measurements

**Filoche, A., Hansen, J., Vinsen, K., Dawson, T. (2026).** *Journal of Geophysical Research:
Machine Learning and Computation*, 3(5), e2026JH001454. DOI: 10.1029/2026JH001454. Published
online 17 September 2026. Crossref gives the full given names as Arthur, Jeff, Kevin and Travis.

## Summary
A deep-learning post-processor for the ECMWF operational wave forecast at one deep-water site in
the Browse Basin, off north-west Australia. An attention-based encoder-decoder, conditioned on
local tidal water level and temporal (date) embeddings, maps the ECMWF **directional spectrum
forecast** to the **1D variance density spectrum measured by a local moored buoy**, across a
5-day forecast horizon. It is trained on 3.5 years of paired forecasts and observations.

Per the abstract, the model improves on ECMWF mainly in the low-frequency swell regime. At a
120 h lead, swell Hs RMSE falls from 0.16 to 0.12 m and swell peak-period error from 2.22 to
1.39 s, and ECMWF's premature swell-arrival bias at this site is mitigated. The network
"struggled to resolve the stochastic variance in higher-frequency wind seas".

The following come from the Zenodo snapshot (its `docs/` and
`results/SpecX_Ensemble/Exp_001/figures/ensemble_rmse_table.tex`) and have **not** been checked
against the published text:
- *Split.* Chronological: training 2020-01-01 to 2022-06-21, validation 2022-07-01 to
  2022-12-22, test calendar year 2023.
- *Partitions.* A fixed frequency cut, not a spectral partitioning: total ≈0.034–0.51 Hz, swell
  0.034–0.10 Hz, wind sea 0.10–0.51 Hz. The 0.10 Hz cut-off was chosen from the local spectral
  climatology.
- *Buoy history.* Five days of buoy history were offered to the hyperparameter search as a
  context input and not selected. The authors' limitations file reads this as the history
  encoder "does not yet extract useful persistence information".
- *ECMWF baseline RMSE on the 2023 test year.* The table prints no units; Hs is in m and the
  periods in s, per the abstract.

| Lead | Total Hs | Total Tp | Total Tm02 | Swell Hs | Swell Tp | Swell Tm02 | Wind-sea Hs | Wind-sea Tp | Wind-sea Tm02 |
|---|---|---|---|---|---|---|---|---|---|
| +3 h | 0.16 | 3.38 | 0.66 | 0.15 | 1.89 | 0.74 | 0.14 | 1.41 | 0.45 |
| +1 d | 0.16 | 3.53 | 0.70 | 0.14 | 2.05 | 0.75 | 0.14 | 1.40 | 0.47 |
| +2.5 d | 0.23 | 3.71 | 0.78 | 0.15 | 2.26 | 0.79 | 0.21 | 1.46 | 0.50 |
| +5 d | 0.25 | 3.79 | 0.92 | 0.16 | 2.22 | 0.81 | 0.24 | 1.47 | 0.59 |

At +5 d the snapshot's ECMWF swell values (0.16 m, 2.22 s) agree with the abstract. Its
ML-ensemble values (swell Hs 0.13 m, swell Tp 1.38 s) do not: the abstract gives 0.12 m and
1.39 s. So the snapshot predates the final text. Cite the paper, not the snapshot, for any ML
number.

## Relevance to this manuscript
High, for three reasons.

*A published physical-model spectral baseline.* It is one of very few published verifications of
an operational numerical model's frequency-spectrum forecast against a buoy by lead time. It is
also the only one found that reports buoy-verified swell and wind-sea errors separately at each
lead: `rogers2025espc` also splits by band, but verifies against satellite spectra. Its Tm02
errors do not change when the spectrum is rescaled, so they stay comparable if the manuscript
reports only the shape forecast. It is *context*, not a head-to-head:
- different site (tropical north-west Australia, vs the southeast Pacific);
- different period (2023, vs 2017);
- different partition method (a fixed 0.10 Hz cut, vs trough-to-trough partitions labelled by
  γ\*).

*It narrows the Introduction's layer-2 gap claim.* This is a learned forecast of the full 1D
spectrum at a buoy over multi-day leads. So "none forecast the full spectrum" (`../../CLAUDE.md`
§0.3) no longer holds without qualification. The distinction that survives is the input: this
paper post-processes a numerical forecast, taking the ECMWF spectrum as input, whereas this
manuscript forecasts from the buoy's own history with no numerical-model input.

*Buoy history was not selected.* Where a numerical spectral forecast is available, buoy history
added nothing to their model. A reviewer may put this to the premise that a buoy's own history
carries forecast skill. It does not contradict that premise, because their setup has the physical
forecast and ours has none. It does frame the comparison: a buoy-history model is most useful
where a numerical forecast is unavailable, too costly, or poorly suited to the site. **This
finding is from the snapshot docs only — confirm it in the full text before citing it.**

A lesser point: their swell-improves, wind-sea-stays-hard result mirrors, from the post-processing
side, the wind-sea/swell distinction that motivates this manuscript's partition-conditioned
diagnostics.

## Suggested use
- **Discussion, physical-baseline paragraph.** Cite for ECMWF's Tm02 and partition-Hs RMSE by
  lead time as published context, alongside the planned same-site numerical-model comparison.
  State the site, period and partition-method caveats in the same sentence.
- **Introduction, the "work that does engage with the spectrum" paragraph.** It fits as a third
  line of prior work, next to the estimation and spatial-forcing lines already cited: learned
  correction of a numerical model's spectral forecast at a station. The existing closing sentence
  ("from a station's own recent spectral history has received considerably less attention")
  still holds with it added.
- **Before quoting any number or the buoy-history finding,** read the full text. It is open
  access; fetch it manually from the DOI and replace the snapshot values with the published
  table.
