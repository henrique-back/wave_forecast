---
key: breunung2023dmd
title: Data-Driven, High Resolution Ocean Wave Forecasting and Extreme Wave Predictions
authors: Breunung, T.; Balachandran, B.
year: 2023
venue: Ocean Engineering, 268, art. 113271
relevance: medium
source_file: none — found via gap-verification websearch, 2026-09-18; no local PDF retrieved
---

# Data-Driven, High Resolution Ocean Wave Forecasting and Extreme Wave Predictions

**Breunung, T., Balachandran, B. (2023).** *Ocean Engineering*, 268, art. 113271.
DOI not located during the websearch pass — omitted from `refs.bib` rather than guessed.

**Provenance note:** found during the Track B gap-verification websearch (2026-09-18) while
checking whether Dynamic Mode Decomposition (DMD) has prior use as an *auxiliary input feature*
for ocean-wave/buoy forecasting (as opposed to DMD being the forecasting method itself). Citation
details come from that search's result snippets, not from reading the paper directly.

## Summary
Decomposes sea-surface elevation into rapid oscillations plus slowly varying amplitudes, then
forecasts those amplitude time series using linear autoregressive models fit via DMD, validated
on wave-tank and ocean-buoy data.

## Relevance to this manuscript
Medium — the closest DMD-adjacent precedent found, but DMD here *is* the forecasting mechanism
(an AR/DMD surrogate applied directly to amplitude series), not a feature extracted and fed as an
auxiliary encoder input alongside a separate primary model — which is how this project's DMD
features are used (see `sections/02_methods.tex`, "Auxiliary dynamical features"). This
distinction is why "DMD as an auxiliary feature" was reported back as a genuinely clean gap in
the gap-verification pass (see `notes.md`, "Gap-verification comparators") even with this paper
as the closest hit.

## Suggested use
Cite when introducing the DMD auxiliary feature in Methods/Discussion, explicitly to draw the
"DMD as forecaster" vs. "DMD as auxiliary feature" distinction rather than leaving the novelty
claim unqualified. Confirm the DOI before final submission — see provenance note above.
