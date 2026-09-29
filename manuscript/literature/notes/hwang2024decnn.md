---
key: hwang2024decnn
title: Applying Artificial Neural Network to Zero-Crossing Wave Parameters for the Wave Spectrum
authors: Hwang, S.; Lee, J.L.; Chun, H.
year: 2024
venue: Ocean Engineering, 309, 118331
relevance: high
source_file: literature/hwang2024decnn.pdf
---

# Applying Artificial Neural Network to Zero-Crossing Wave Parameters for the Wave Spectrum

**Hwang, S., Lee, J.L., Chun, H. (2024).** *Ocean Engineering*, 309, 118331.
DOI: 10.1016/j.oceaneng.2024.118331.

## Summary
A deconvolutional neural network (DeCNN) trained on Korean buoy data reconstructs the full wave
energy spectrum from zero-crossing-derived scalar parameters (Hmax, Tmax, H1/3, T1/3, Hm, Tz),
using a KL-divergence loss (plus a relaxation term). Shown to generalize to other Korean seas
and to outperform JONSWAP-based parametric spectra in some respects.

## Relevance to this manuscript
High — an ML model that outputs the full spectrum shape and explicitly uses a KL-divergence
loss term for spectral comparison, directly analogous to one term of this project's composite
loss.

## Suggested use
Strong candidate citation for the composite-loss design paragraph (precedent for KL-divergence
as a spectral loss term), and as a prior-art comparator for spectrum-shape prediction from
scalar/derived parameters, contrasting with this project's frequency-resolved autoregressive
input.
