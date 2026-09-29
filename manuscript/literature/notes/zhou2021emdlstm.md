---
key: zhou2021emdlstm
title: Improving Significant Wave Height Forecasts Using a Joint Empirical Mode Decomposition-Long Short-Term Memory Network
authors: Zhou, S.; Bethel, B.J.; Sun, W.; Zhao, Y.; Xie, W.; Dong, C.
year: 2021
venue: Journal of Marine Science and Engineering, 9(7), 744
relevance: high
source_file: literature/zhou2021emdlstm.pdf
---

# Improving Significant Wave Height Forecasts Using a Joint Empirical Mode Decomposition-Long Short-Term Memory Network

**Zhou, S., Bethel, B.J., Sun, W., Zhao, Y., Xie, W., Dong, C. (2021).**
*Journal of Marine Science and Engineering*, 9(7), 744. DOI: 10.3390/jmse9070744.

## Summary
Couples Empirical Mode Decomposition (EMD) with an LSTM to forecast Hs at 3-72h horizons from
two NDBC buoys (Atlantic, near the Bahamas, 2018-2019). EMD-LSTM outperforms a plain LSTM,
especially beyond 24h, though the LSTM still struggles with high-frequency components.

## Relevance to this manuscript
High — ML (LSTM) Hs forecasting from NDBC buoy data at lead times directly comparable to this
project's own 6/12/24/48h grid; also a signal-decomposition-as-auxiliary-feature precedent
relevant to justifying DMD as an auxiliary dynamical feature (parallel rationale, different
decomposition).

## Suggested use
Prior Hs-forecasting-with-ML baseline to contrast against; candidate citation motivating
decomposition-based auxiliary features (parallel to this project's DMD features), and for noting
a known LSTM weakness on high-frequency spectral content.
