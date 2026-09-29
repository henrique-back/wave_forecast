---
key: cooper2023thesis
title: Applications of the Bures-Wasserstein Distance in Linear Metric Learning
authors: Cooper, Davis
year: 2023
venue: PhD thesis, Victoria University of Wellington
relevance: low
source_file: literature/cooper2023thesis.pdf
---

# Applications of the Bures-Wasserstein Distance in Linear Metric Learning

**Cooper, Davis (2023).** PhD thesis, Victoria University of Wellington.

## Summary
Develops linear/low-rank metric-learning algorithms using the Bures-Wasserstein distance
(a matrix/Gaussian optimal-transport metric) for classification tasks on feature/image/SPD-matrix
data. No wave or spectral content.

## Relevance to this manuscript
Low — unrelated domain. Importantly, the Wasserstein formulation used in this project's
composite loss is a 1D distributional Wasserstein-2 (closed-form quantile-domain), not the
Bures-Wasserstein *matrix* metric this thesis develops — the two are mathematically distinct
objects that happen to share a name.

## Suggested use
Not recommended as a citation for the composite-loss justification.
`decisions/wasserstein_kl_justification.tex` already cites Cuturi (2013) and Peyré & Cuturi
(2019) for the relevant 1D optimal-transport background — this thesis does not add to that.
Likely excluded from the manuscript entirely.
