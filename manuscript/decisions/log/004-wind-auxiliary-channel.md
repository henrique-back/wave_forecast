---
status: kept, marginal
date: 2026-07-06
commits: [9c6f94d, fe5f666]
category: data
---

# 004 — Add `wind_u`/`wind_v` as an encoder-only auxiliary input

## Context

Wave spectra are wind-forced, so local wind observations were hypothesized to carry
predictive signal beyond the spectrum's own recent history — particularly for the
wind-sea part of mixed sea states. NDBC `wind.txt` (stdmet format) provides wind
direction/speed at the same buoy.

## Change

`utils/data_processing.py::process_wind()` converts NDBC `WDIR`/`WSPD` (masking sentinels
`WDIR==999`, `WSPD==99.0` to NaN) into `wind_u`/`wind_v` components **before**
time-interpolation, since interpolating the raw circular angle across a gap would pass
through the wrong side of the compass. `nn/prepare_aux.py` windows these into
`(samples, seq_len, num_aux_channels)`, fused additively into the encoder token via a
separate `aux_embedding` (`nn/transformer.py`) — never reaching the decoder, since wind is
never a forecast target. Selected via `aux_set='wind'` (`nn/channels.py`).

## Evidence

`results/RESEARCH_LOG.md`, `shape` target, `weighted_mean_SS`-objective studies at 6h/12h:

| | `aux_set` | `SS` 6h | `SS` 12h |
|---|---|---:|---:|
| `hs_shape_v6` | none | 0.1709 | 0.1821 |
| `wind_combined_v7` | wind | 0.1975 | 0.1900 |

A small, consistent improvement at both lead times shown, but this is not an isolated
single-variable ablation — `wind_combined_v7` is a later study (`n_warmup_steps` change from
[[005]] already landed) and the comparison is against `hs_shape_v6`'s `shape`-only numbers,
not a paired `aux_set=none` vs. `aux_set=wind` run on an otherwise identical config. The
`hs` target's `SS` in the same two studies (0.068/0.062 → 0.026/0.048) is more equivocal
— wind slightly *hurts* the 6h `hs` forecast in this comparison, which is worth resolving
with a cleaner ablation before this is cited in the manuscript as a settled result.

## Decision

Kept for `shape`/`density` targets (default config for later experiments uses
`aux_set='wind'`), but flagged as marginal — the mixed signal on `hs` suggests this hasn't
been rigorously isolated. Treat as "wind helps the shape forecast, unclear for `hs`" rather
than a clean win, unless a dedicated paired ablation is run before writing this up.

## Related

Code: `utils/data_processing.py::process_wind`, `nn/prepare_aux.py`, `nn/channels.py`
Other decisions: [[005]]
Manuscript section this might feed: Methods — auxiliary inputs. Do not cite the `SS` deltas
above as a clean ablation result without first running (or finding) a paired same-study
`aux_set` comparison.
