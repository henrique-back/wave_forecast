---
key: james2018wave
title: A Machine Learning Framework to Forecast Wave Conditions
authors: James, S.C.; Zhang, Y.; O'Donncha, F.
year: 2018
venue: Coastal Engineering, 137, 1-10
relevance: high
source_file: none — found via find-reference websearch, 2026-09-21; read via arXiv:1709.08725 (preprint of the published version), no local PDF retrieved
---

# A Machine Learning Framework to Forecast Wave Conditions

**James, S.C., Zhang, Y., O'Donncha, F. (2018).** *Coastal Engineering*, 137, 1-10.
DOI: 10.1016/j.coastaleng.2018.03.004.

## Summary
Trains machine learning models (supervised regression) on many thousands of iterations of a
SWAN physics-based wave model (Monterey Bay test site, forced by measured wave conditions,
ocean-current nowcasts, and reported winds) to reproduce spatially variable Hs and characteristic
period. The trained ML surrogate reproduces Hs with 9 cm RMSE and correctly identifies over 90%
of characteristic periods, while requiring "only a fraction (< 1/1,000th) of the computation
time compared to forecasting with the physics-based model" (abstract, verbatim). The
Introduction states explicitly: "Computational expense is often a major limitation of real-time
forecasting systems," motivating the ML-surrogate approach.

## Relevance to this manuscript
High for the motivating "why AI/DL for wave forecasting" argument specifically — a
directly-quotable, peer-reviewed statement that numerical (physics-based) wave models are
computationally expensive enough to limit real-time/operational forecasting, and that a trained
ML model can reproduce their output at a small fraction of the compute cost. Lower relevance to
the modelling approach itself: this is a spatial surrogate for a numerical model's *output*
(trained on model iterations, not on buoy history autoregressively), not a spectrum-forecasting
model trained directly on observed buoy time series, so it should not be cited as a
methodological precedent for this project's architecture.

## Suggested use
Introduction/motivation citation for the computational-cost argument in the "usefulness of
AI/DL models" framing (see `personal_notes.md`) — pairs with `gao2025jgr`, which makes the same
"cheap DL surrogate for a numerical wave model" point for directional spectra specifically.
