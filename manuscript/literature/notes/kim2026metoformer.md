---
key: kim2026metoformer
title: A Multi-Scale and Horizon-Adaptive Transformer for Operational Metocean Forecasting in Korean Coastal Buoy Networks
authors: Kim, Y.; Kwon, C.; Lee, H.; Seok, J.
year: 2026
venue: Ocean Engineering, 362, 126344
relevance: high
source_file: literature/kim2026metoformer.pdf
---

# A Multi-Scale and Horizon-Adaptive Transformer for Operational Metocean Forecasting in Korean Coastal Buoy Networks

**Kim, Y., Kwon, C., Lee, H., Seok, J. (2026).** *Ocean Engineering*, 362, 126344.
DOI: 10.1016/j.oceaneng.2026.126344.

## Summary
Proposes "MetoFormer-HQ", a PatchTST-based Transformer with multi-scale patch embedding, gated
channel interaction, and horizon-adaptive decoding, for joint multi-step/multi-variable (SWH,
wind, current, etc.) forecasting from a Korean coastal buoy network. Reports RMSE/MAE gains over
PatchTST and N-HiTS baselines, plus diagnostics (persistence, autoregressive, lag-sensitivity
checks) showing short-horizon SWH forecasts are strongly persistence-dominated.

## Relevance to this manuscript
High — the closest architectural analog found in the local corpus: Transformer-based,
multi-horizon, buoy-derived wave/metocean forecasting, and it explicitly grapples with the
persistence-baseline issue this project's Skill Score metric is built around. It forecasts
scalar SWH and other bulk variables jointly, not the full directional spectrum, and uses
patch-based (not frequency-structured per-bin) embedding.

## Suggested use
Prime recent-baseline citation to contrast architecture choices against (patch-based Transformer
vs. this project's frequency-structured per-bin embedding) in the Introduction/Discussion, and to
support the persistence-dominance concern already reflected in this project's Skill-Score-based
evaluation design. Also the closest match found for gap-verification topic "transformer
forecasting a wave-related target from buoy history" (see notes.md, Gap-verification section).
