---
key: benbouallegue2024rise
title: "The Rise of Data-Driven Weather Forecasting: A First Statistical Assessment of Machine Learning-Based Weather Forecasts in an Operational-Like Context"
authors: Ben Bouallègue, Z.; Clare, M.C.A.; Magnusson, L.; Gascón, E.; Maier-Gerber, M.; Janoušek, M.; Rodwell, M.; Pinault, F.; Dramsch, J.S.; Lang, S.T.K.; Raoult, B.; Rabier, F.; Chevallier, M.; Sandu, I.; Dueben, P.; Chantry, M.; Pappenberger, F.
year: 2024
venue: Bulletin of the American Meteorological Society, 105(6), E864-E883
relevance: high
source_file: literature/benbouallegue2024rise.pdf — VERSION MISMATCH, this is the arXiv v2 preprint (16 pp), not the published BAMS article; cite the BAMS version, verify quotes against it
---

# The Rise of Data-Driven Weather Forecasting

**Ben Bouallègue, Z., Clare, M.C.A., Magnusson, L., et al. (2024).** *Bulletin of the American
Meteorological Society*, 105(6), E864-E883. DOI: 10.1175/BAMS-D-23-0162.1.

## Summary
ECMWF's own head-to-head assessment of a machine-learning global weather model (PanguWeather)
against the operational IFS, run from identical initial conditions in an operational-like
setting. Reported conclusion: "The results are very promising, with comparable accuracy for both
global metrics and extreme events, when verified against both the operational IFS analysis and
synoptic observations."

Critically, it also names the ML model's drawbacks explicitly: **overly smooth forecasts**,
increasing bias with forecast lead time, and poor performance in predicting tropical cyclone
intensity.

## Relevance to this manuscript
High, and more useful than it first appears — for **two separate reasons**.

1. *The motivation claim, honestly bounded.* This is the strongest available operational
   evidence for data-driven weather forecasting, and it supports "comparable to" or "competitive
   with" — **not** "consistently outperforms". The Introduction's opening claim must be worded
   accordingly (see `personal_notes.md`, "weather - numerical vs AI models").

2. *The over-smoothing parallel — the more valuable point.* This paper independently documents
   **over-smoothing as a characteristic failure mode of data-driven forecasting models**, in a
   completely different geophysical domain, verified operationally by a national forecasting
   centre. That is exactly the failure mode this manuscript identifies in learned wave-spectrum
   forecasts and builds its composite loss and peak-resolved evaluation panel to counter. It
   turns this manuscript's central problem from a wave-specific quirk into an instance of a
   documented, cross-domain pathology of data-driven forecasting — a substantially stronger
   framing than "our model blurs peaks."

Atmosphere-only: says nothing about ocean or wave forecasting.

## Suggested use
Introduction, twice. First in the opening paragraph with `rasp2024weatherbench2`, for the
data-driven-vs-NWP state of play, worded as "comparable/competitive", never "outperforms".
Second, and more importantly, in the paragraph motivating the over-smoothing problem — as
independent, cross-domain evidence that learned forecasters tend to produce over-smooth fields,
which is why a forecast evaluated only on aggregate error can look good while being physically
wrong.
