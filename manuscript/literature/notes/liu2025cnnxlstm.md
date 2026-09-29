---
key: liu2025cnnxlstm
title: Regional Wave Spectra Prediction Method Based on Deep Learning
authors: Liu, Y.; Li, R.; Hu, W.; Ren, P.; Xu, C.
year: 2025
venue: Journal of Marine Science and Engineering, 13(8), 1461
relevance: high
source_file: literature/liu2025cnnxlstm.pdf
---

# Regional Wave Spectra Prediction Method Based on Deep Learning

**Liu, Y., Li, R., Hu, W., Ren, P., Xu, C. (2025).** *Journal of Marine Science and
Engineering*, 13(8), 1461. DOI: 10.3390/jmse13081461.

## Summary
A CNN + xLSTM model predicts gridded 1D frequency wave spectra over a regional domain
(Bohai/Yellow Sea) from ERA5 wind/pressure fields, using a frequency-weighted loss to sharpen
high-frequency spectral prediction, validated against ERA5 and three NDBC-style buoys.

## Relevance to this manuscript
High — deep-learning spectral (not just Hs) forecasting that explicitly uses a
frequency-weighted loss to improve high-frequency spectral fidelity, directly relevant to
justifying this project's trapezoidal frequency-weighted loss term and its E(f) shape-target
framing. Also a candidate density-forecasting ML baseline, though regional/wind-driven rather
than single-point autoregressive.

## Suggested use
Candidate citation for the frequency-weighted spectral loss design rationale (Eq. for
$\mathcal{L}_{\text{bin}}$ in `02_methods.tex`); contrasting ML spectrum-forecasting baseline
(regional/wind-driven vs. this project's single-point autoregressive spectrum-to-spectrum
approach).
