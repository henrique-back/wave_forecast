---
status: kept
date: 2026-07-14
commits: [6aadff5, 7176524]
category: architecture
---

# 006 — `FreqDimEmbedding` internals: flatten+`Linear` pool → dilated-conv + attention pool

## Context

The first version of `FreqDimEmbedding` (see [[002]]) still collapsed the per-bin
representations into one token via a flatten+`Linear` step — i.e. a fixed, position-indexed
weighting of the 47 bins, with no notion of *which* bin was which beyond its index in the
flattened vector, and no way for the pooling to adapt to where the spectral peak currently
sits.

## Change

Per commit `6aadff5`: each bin now gets an explicit frequency-identity signal (a sinusoidal
encoding of its actual log-frequency value, wavelengths scaled to the grid's real span,
plus a learned zero-initialized residual) before aggregation. Aggregation itself changed
from flatten+`Linear` to a dilated-conv frontend (`TemporalConvFrontend` reused along the
frequency axis, letting each bin absorb local context from neighbors) followed by a
single-query attention pool (`_FreqAttentionPool`) that collapses all bins into one
`embed_dim` token with content-aware weighting. Commit `7176524` (same day) applied this to
the decoder's `FreqDimEmbedding` instance as well, plus set `norm_first=True` on
`nn.Transformer`.

## Evidence

No isolated ablation run exists for this refinement specifically — `results/RESEARCH_LOG.md`
has no study bracketing exactly this commit with everything else held fixed; the closest
studies (`shape_v8` → `shape_v9` → `shape_v10`) each bundle this change together with
RMSE-weighting fixes, positivity constraints (clamp/softplus), and later a new input channel
(`r2`) and optimizer switch, so no single-variable delta can be attributed to this change
alone from `results/` data.

## Decision

Kept, on architectural reasoning rather than an isolated metric delta: a position-blind
flatten+`Linear` pool cannot track a moving spectral peak, while content-aware attention
pooling can. This is also the mechanism later cited in root `CLAUDE.md`'s comparison against
the Meta 4 per-frequency architecture proposal (the doc's suggested frequency-attention and
frequency-convolution `Ψ` mechanisms are both already implemented here, combined rather than
picking one).

## Related

Code: `nn/freq_embedding.py::FreqDimEmbedding`, `nn/freq_embedding.py::_FreqAttentionPool`
Other decisions: [[002]]
Manuscript section this might feed: Methods — model architecture (the `FreqDimEmbedding`
paragraph); useful context if a reviewer asks why frequency identity is injected additively
rather than concatenated, per the Meta 4 proposal comparison in root `CLAUDE.md`.
