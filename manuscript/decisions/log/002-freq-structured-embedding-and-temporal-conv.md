---
status: kept
date: 2026-07-03
commits: [d9e125b]
category: architecture
---

# 002 — Encoder front-end: flat `Linear` embedding → `FreqDimEmbedding` + `TemporalConvFrontend`

## Context

The original baseline (`baseline_v2`) flattened each timestep's `(num_freqs, num_channels)`
spectrum through one large `Linear` layer (`nn/embedding.py::Embedding`) with a standard
Transformer encoder-decoder on top — no structure exploiting the fact that the 47 frequency
bins are a smooth, non-uniformly (log-)spaced curve, and no local-temporal-pattern extraction
before the global self-attention layers.

## Change

Introduced `nn/freq_embedding.py::FreqDimEmbedding` (per-bin linear projection, frequency
identity signal, then pooled into one `embed_dim` token per timestep) for the encoder, and
`nn/temporal_conv.py::TemporalConvFrontend` (three dilated `Conv1d` layers, dilations 1/2/4)
ahead of the Transformer's self-attention layers to extract local 3h/5h/9h temporal patterns
first. Both are described in full in the root `CLAUDE.md` "Model" section.

## Evidence

`results/RESEARCH_LOG.md`: `baseline_v2` (flat `Linear`, no conv, `density` target,
`Shape_RMSE`-objective study) — `SS` 0.157 / 0.175 / 0.192 at 6h/12h/24h. The first study
built on the new front-end, `weightedmeanSS_conv_freqemb_v3` (`density` target) — `SS` 0.220 /
0.221 / 0.229 at 6h/12h/24h — a clear improvement at every lead time.

This is **not** a clean single-variable ablation: the objective metric also changed between
these two studies (`Shape_RMSE` → `weighted_mean_SS`, see [[001]]), so part of the SS
difference could reflect what the search was optimizing for, not only the architecture
change. No isolated "same objective, front-end on vs. off" run exists in `results/`.

## Decision

Kept. The directional improvement is large and consistent across all three lead times, and
the architectural rationale (log-frequency-aware, content-weighted pooling vs. a
position-blind flatten) is not contingent on the objective-metric confound — but the
confound should be disclosed if this comparison is ever quoted with exact numbers in the
manuscript.

## Related

Code: `nn/freq_embedding.py`, `nn/temporal_conv.py`, `nn/transformer.py`
Other decisions: [[001]] (the confounded objective-metric change), [[006]] (later refinement
of `FreqDimEmbedding`'s internal pooling mechanism)
Manuscript section this might feed: Methods — model architecture; Results — ablation summary
table (flag the confound if citing these exact SS numbers).
