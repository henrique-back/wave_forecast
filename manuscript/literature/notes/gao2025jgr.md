---
key: gao2025jgr
title: Physics-Guided Deep Learning for Modeling Single-Point Wave Spectra Using Wind Inputs of Two Resolutions
authors: Gao, T.; Jiang, H.
year: 2025
venue: "JGR: Machine Learning and Computation, 2, e2024JH000492"
relevance: medium
source_file: literature/gao2025jgr.pdf
---

# Physics-Guided Deep Learning for Modeling Single-Point Wave Spectra Using Wind Inputs of Two Resolutions

**Gao, T., Jiang, H. (2025).** *JGR: Machine Learning and Computation*, 2, e2024JH000492.
DOI: 10.1029/2024JH000492.

## Summary
A deep-learning model predicts single-point directional wave spectra (2D, frequency x direction)
from dual-resolution wind fields (local high-res for wind-sea, basin-scale low-res for remote
swell), trained/evaluated against ERA5 and IOWAGA hindcasts at four Pacific/coastal points,
aiming to surrogate numerical wave models cheaply.

## Relevance to this manuscript
Medium — ML forecasting of directional wave spectra at a point, but the input is wind fields
(not prior spectra autoregressively) and the target is a spatial/multi-resolution wind-driven
spectrum — a different problem framing than this project's spectrum-to-spectrum forecasting.

## Suggested use
Could support the gap statement, or motivate physical wind-sea vs. swell separation as a
modelling principle generally; a contrasting related-work citation for directional-wave-spectrum
DL surrogates, not a direct baseline.
