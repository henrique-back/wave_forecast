---
key: jiang2024comment
title: "Comment on Papers Using Machine Learning for Significant Wave Height Time Series Prediction: Complex Models Do Not Outperform Auto-Regression"
authors: Jiang, H.; Zhang, Y.; Qian, C.; Wang, X.
year: 2024
venue: Ocean Modelling, 189, 102364
relevance: medium-high
source_file: literature/jiang2024comment.pdf
---

# Comment on Papers Using Machine Learning for Significant Wave Height Time Series Prediction: Complex Models Do Not Outperform Auto-Regression

**Jiang, H., Zhang, Y., Qian, C., Wang, X. (2024).** *Ocean Modelling*, 189, 102364.
DOI: 10.1016/j.ocemod.2024.102364.

**Correction note:** the locally-held file is an accepted-manuscript PDF whose filename pattern
suggested *Environmental Modelling & Software*; a dedicated websearch (Crossref DOI + ScienceDirect
lookup, 2026-09-18) confirmed the actual venue is **Ocean Modelling**, vol. 189, art. 102364. Use
that venue, not the one implied by the filename.

## Summary
A comparative/critical study across 16 NDBC buoys (2016-2017 — the same buoy era this project's
own record spans) showing AR, XGBoost, ANN, LSTM, and WaveNet perform nearly identically for
univariate Hs forecasting (1-72h horizons) — i.e. complex DL models mostly just learn linear
autocorrelation — and flags a methodological error (signal decomposition applied across the
train/test boundary) common in the literature.

No direct rebuttal was found (checked via Semantic Scholar's citing-works list, 2026-09-18); most
papers that cite it treat it as a caution rather than contest it. One 2026 paper, "On the limits
of univariate deep learning for significant wave height forecasting" (title/venue not yet fully
verified — see notes.md "Pending verification"), appears to reinforce rather than rebut its
conclusion.

## Relevance to this manuscript
Medium-high — not a comparable method, but a cautionary/critical baseline paper directly
relevant to this project's own evaluation philosophy (Skill Score vs. persistence).

## Suggested use
Use in the Introduction's gap framing (justifying rigorous persistence/AR baselines rather than
assuming ML wins by default) and explicitly in the Discussion — this project's own Skill-Score
-vs-persistence result should be read against this paper's critique, engaging with it directly
rather than only citing it in passing.
