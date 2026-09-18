---
status: kept
date: 2026-07-21
commits: [a541c6e]
category: architecture
---

# 011 — `FreqDimEmbedding`'s frequency-axis conv: zero-padding → replicate-padding

## Context

`FreqDimEmbedding`'s frequency-axis dilated conv (`nn/freq_embedding.py`, reusing
`TemporalConvFrontend` along the frequency axis instead of time) used zero-padding at the
grid boundary (0.02 Hz / 0.485 Hz edges), like a normal `Conv1d`. This fabricates fake
zero-energy bins just outside the real grid, which corrupted the ~7 bins nearest each edge
— visible as a spurious low-frequency bump/negative dip in `shape_v8`'s test-set
predictions.

## Change

Switched the frequency-axis conv's padding mode to replicate-padding (the edge bin's real
value is repeated as padding instead of zero), matching physical intuition — the spectrum
doesn't actually go to zero just outside the measured band. `STUDY_VERSION` bumped v8→v9
in `scripts/optimize.py` (architecture change, so v8 trials are not comparable to v9's).

## Evidence

Bug-driven fix: the artifact was diagnosed visually in `shape_v8`'s test-set inference
plots (spurious bump/dip near the grid edges), not from a quantitative before/after
comparison — no isolated `metrics.json` isolates this one change from the rest of the
v8→v9 transition. `results/shape_v9/` is the first study run with the fix.

## Decision

Kept. The physical argument (real spectra don't discontinuously drop to zero at the
measurement band's edge) is independent of any numeric result, and the visual artifact it
fixes was clearly a correctness bug, not a modeling trade-off.

## Related

Code: `nn/freq_embedding.py`, `nn/temporal_conv.py::TemporalConvFrontend`
Other decisions: [[002]], [[006]] (`FreqDimEmbedding`/`TemporalConvFrontend` internals)
Manuscript section this might feed: Methods — model architecture (implementation detail,
likely not manuscript-worthy on its own per the parent CLAUDE.md's "no code-shaped framing"
rule — mention only if the boundary-handling choice itself needs justifying).
