---
status: kept
date: 2026-08-03
commits: [34fa691]
category: data
---

# 019 — Dynamic Mode Decomposition (DMD) features as a new `'dmd'` auxiliary channel

## Context

The encoder had to infer implicitly, from raw historical spectra alone, whether the
currently-observed wind-sea/swell systems were growing or decaying. [[004]] had already
established the pattern of feeding a scalar-per-timestep auxiliary channel (there, wind)
into the encoder separately from the frequency-resolved input; DMD was explored as a
second such channel giving the model this growth/decay information directly instead of
requiring it to be learned implicitly.

## Change

Added `nn/prepare_dmd.py::compute_dmd_features`: DMD fits a linear operator A such that
`x_{t+1} ≈ A x_t` across a sample's `seq_len` window of (already-windowed,
already-normalized) density spectra, then decomposes A into modes with complex
eigenvalues — each eigenvalue's magnitude gives a growth/decay rate, its phase gives an
oscillation frequency. The dominant `n_modes=4` modes' `(growth_rate, frequency,
amplitude)` triples are exposed as a new `'dmd'` entry in `AUX_CHANNEL_SETS`
(`nn/channels.py`), broadcast across `seq_len` to match `nn/prepare_aux.py`'s output
contract (DMD needs a *different* preparation function than `prepare_aux` because it needs
each sample's already-windowed history first, to fit DMD on that sample's own history,
whereas `prepare_aux` windows an already-fully-computed per-timestep series). Normalized
(not physical) density is fine as DMD's input, since DMD eigenvalues are invariant to a
fixed per-bin linear rescaling (a similarity transform) — only mode amplitude becomes
"relative to normalized units," still a meaningful feature. `AUX_SET` switched from
`'none'` to `'dmd'` as the new default.

## Evidence

**No committed before/after metrics file exists for this change in isolation.** The code
comment describing it (`scripts/optimize.py`'s `STUDY_VERSION` v12 entry) states it was
"manually validated alone (smaller, less consistent effect than the Wasserstein term,
[[020]]) and in combination with it (best `Peak_Separation_Recall`/`Peak_Count_Pred_Mean`
of anything tested, and the visually sharpest/tallest peaks across every known-multimodal
sample checked, even though whole-spectrum `Shape_RMSE` slightly favored the
Wasserstein-only run)" — this is a manual, non-committed comparison; no `metrics.json` or
`results/` artifact isolating DMD's own contribution was found. Per this log's own sourcing
rules, this is disclosed explicitly rather than presented as a clean ablation result.

## Decision

Kept as the new default `AUX_SET`, on the strength of the manual comparison described
above, not a committed quantitative ablation. If a future study needs to justify this
choice quantitatively (e.g. for the manuscript), an isolated `aux_set='none'` vs.
`aux_set='dmd'` run at matched architecture/loss should be run first — none currently
exists.

## Related

Code: `nn/prepare_dmd.py`, `nn/channels.py` (`_DMD_COLUMNS`, `AUX_CHANNEL_SETS`)
Other decisions: [[004]] (established the auxiliary-channel pattern this reuses), [[020]]
(bundled in the same commit — validated together against the same manual comparison)
Manuscript section this might feed: Methods — auxiliary input features. Per the parent
CLAUDE.md §6, must be introduced by its physical/statistical function (features
characterising the growth/decay dynamics of the recently observed spectral time series via
Dynamic Mode Decomposition), not by its code role.
