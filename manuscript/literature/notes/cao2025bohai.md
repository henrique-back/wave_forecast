---
key: cao2025bohai
title: The Method of Single Points Wave Spectrum Generation for Regional Sea Based on Multimodal Model in the Bohai Sea, China
authors: Cao, X.; Zuo, M.; Wang, X.; Wang, Z.; Niu, Y.; Wang, P.; Lv, Y.
year: 2025
venue: Dynamics of Atmospheres and Oceans, 111, 101586
relevance: medium-high
source_file: literature/cao2025bohai.pdf
---

# The Method of Single Points Wave Spectrum Generation for Regional Sea Based on Multimodal Model in the Bohai Sea, China

**Cao, X., Zuo, M., Wang, X., Wang, Z., Niu, Y., Wang, P., Lv, Y. (2025).**
*Dynamics of Atmospheres and Oceans*, 111, 101586. DOI: 10.1016/j.dynatmoce.2025.101586.

## Summary
An encoder-decoder deep-learning model with Coordinate Attention generates full single-point
wave spectra in a semi-enclosed sea from multi-scale wind-field and bathymetry inputs, trained on
WAVEWATCH III-simulated data and validated against buoy Hs. Evaluates via spectrum shape and
bulk-parameter correlation/MAE/RMSE.

## Relevance to this manuscript
Medium-high — a relevant precedent for DL prediction of the full spectrum (not just Hs) with a
spectrum+bulk-parameter evaluation philosophy, but predicts spectra from wind/bathymetry rather
than forecasting from prior buoy-observed spectra, and has no directional (alpha/r) channels or
attention-pooled frequency embedding.

## Suggested use
Supports the claim that full-spectrum (not just scalar Hs) DL generation is a nascent but
limited literature; a contrast point for this project's temporal-autoregressive, buoy-only,
directionally-resolved approach.
