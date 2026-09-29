---
key: sun2025gpgnn
title: "G-PGNN: A Physics-Guided Generative Neural Network Model for Discrete Spectrum Prediction"
authors: Sun, L.; Wang, J.; Li, Z.; Jiao, Z.; Ma, Y.
year: 2025
venue: Ocean Engineering, 340, 122411
relevance: high
source_file: literature/sun2025gpgnn.pdf
---

# G-PGNN: A Physics-Guided Generative Neural Network Model for Discrete Spectrum Prediction

**Sun, L., Wang, J., Li, Z., Jiao, Z., Ma, Y. (2025).** *Ocean Engineering*, 340, 122411.
DOI: 10.1016/j.oceaneng.2025.122411.

## Summary
A generative CNN (encoder-decoder with residual blocks) predicts full 1D discrete wave spectra
directly from ship motion response signals, trained with a multi-term physics-guided loss
(zeroth-moment/energy, peak-frequency-deviation, first-moment, cosine-similarity/non-overlap
-energy terms) combined via a dynamic adaptive weighting scheme. Outperforms CNN/RF/ET
regression baselines on simulated data, especially for spectral shape fidelity.

## Relevance to this manuscript
High — a directly analogous composite/physics-guided multi-term spectral loss design (including
an explicit peak-frequency term), and a full-spectrum (not just Hs) generative prediction target.

## Suggested use
Strong candidate citation and comparator for the composite spectral loss paragraph
(KL/Wasserstein/peak-height design) — precedent for combining multiple physically motivated
spectral loss terms with adaptive weighting. Also supports framing full-spectrum shape fidelity
as a known failure point of naive per-bin losses. Note the input modality differs (ship motion,
not a buoy's own spectral history), so this is a loss-design precedent, not a forecasting-task
precedent.
