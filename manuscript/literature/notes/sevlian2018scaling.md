---
key: sevlian2018scaling
title: A scaling law for short term load forecasting on varying levels of aggregation
authors: Sevlian, R.; Rajagopal, R.
year: 2018
venue: International Journal of Electrical Power & Energy Systems, 98, 350-361
relevance: medium
source_file: literature/sevlian2018scaling.pdf — VERSION MISMATCH, this is the arXiv:1404.0058 v3 preprint of the IJEPES article (ScienceDirect blocked); cite the journal version
---

# A scaling law for short term load forecasting on varying levels of aggregation

**Sevlian, R., Rajagopal, R. (2018).** *International Journal of Electrical Power & Energy
Systems*, 98, 350-361. DOI: 10.1016/j.ijepes.2017.10.032.

## Summary
Derives an empirical scaling law relating short-term electricity load-forecast error to the
level of customer aggregation. The part relevant here is its day-ahead forecaster (model M6),
which uses exactly the shape/magnitude factorisation this manuscript applies to wave spectra.
Quoted verbatim from the local PDF (verified, not recalled):

> "The forecaster works by predicting the daily total consumption p̂_d ∈ R and normalized daily
> shape pattern û_d ∈ R^24 separately. The final prediction x̂_{d+1} = p̂_{d+1} û_{d+1} is the
> product of each individual forecast."

Total consumption is forecast by ARMAX, the normalised shape by a vector ARMAX.

## Relevance to this manuscript
Medium — but important for **honest novelty framing**, which is why it is filed despite being
from an unrelated field.

The scalar-total × normalised-shape factorisation, recombined multiplicatively at inference, is
**not novel in forecasting generally**: it is established practice in short-term electricity
load forecasting. The manuscript should therefore not present the decomposition itself as an
invention. What does appear novel, after a dedicated search that found no ocean-domain
precedent, is its *application to wave-spectrum forecasting* — forecasting a unit-area spectrum
separately from Hs, with the physical m₀ = (Hs/4)² relation supplying the magnitude, and with
the specific motivation of countering spectral peak under-prediction.

Differences worth stating if cited: in Sevlian & Rajagopal the split is a baseline component
rather than a claimed contribution; there is no argument that it fixes peak under-prediction;
the models are linear (ARMAX), not learned autoregressive encoders; and the normalised curve is
load over hour-of-day, so there is no conserved-energy or physical-moment interpretation.

Related framing note: non-dimensionalising spectra by Hs is routine in wave *characterisation*
(the JONSWAP-family parameterisations do it), so novelty should be claimed specifically for
using the unit-area spectrum as a **decoupled forecast target**, not for the normalisation.

## Suggested use
Introduction or Discussion, where the shape/magnitude decomposition is introduced — cite as the
acknowledged precedent from another forecasting domain ("a factorisation established in
short-term load forecasting"), so the contribution is framed as a transfer plus a physical
grounding rather than an invention. Pair with `guo2026loadshape`.
