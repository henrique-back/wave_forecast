---
key: namekar2006ann
title: Application of Artificial Neural Network Model in Estimation of Wave Spectra
authors: Namekar, S.; Deo, M.C.
year: 2006
venue: Journal of Waterway, Port, Coastal, and Ocean Engineering, 132(5), 415-418
relevance: high
source_file: literature/namekar2006ann.pdf
---

# Application of Artificial Neural Network Model in Estimation of Wave Spectra

**Namekar, S., Deo, M.C. (2006).** *Journal of Waterway, Port, Coastal, and Ocean Engineering*
(ASCE), 132(5), 415-418. DOI: 10.1061/(ASCE)0733-950X(2006)132:5(415).

## Summary
A feedforward back-propagation ANN (2 hidden layers) maps scalar Hs/Tz to a 1D spectral density
S(f) at 30 frequencies, using NDBC buoy 42001 data (721 samples, 85/15 split). Outperforms
parametric PM/JONSWAP/Scott's spectra in shape fidelity but underestimates the spectral peak
(corrected post hoc with an empirical multiplier).

## Relevance to this manuscript
High — the earliest ML precedent found for the exact "scalar params -> spectrum shape" problem
that this project's shape/magnitude split addresses from the forecasting side.

## Suggested use
Prior ML baseline for spectrum estimation; the observed peak-underestimation bias directly
motivates the shape/magnitude decomposition and the soft-peak-height loss term in the
Introduction/Discussion — this is a citable, decades-old precedent for exactly the failure mode
this project's composite loss is designed to correct.
