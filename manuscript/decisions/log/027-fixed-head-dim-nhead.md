---
status: kept
date: 2026-09-11
commits: [5586453]
category: training
---

# 027 — `head_dim=32`/`nhead=8` fixed (not searched) for the `shape` target, v13

## Context

`head_dim`/`nhead` had been part of the Optuna search space across every `shape`-target
study since `shape_v10` (`freq_embed_dropout`/`embed_dropout` search-space shape). Adding
`kl_loss_weight`/`peak_loss_weight` as new tunable dimensions for v13 (see [[026]]) would
bring the total to 13 dimensions; before doing so, a cross-version tally of every completed
`shape`-target study sharing that search-space shape was done to see whether any
architecture dimension had already converged to a stable value worth fixing.

## Change

`FIXED_HEAD_DIM=32`, `FIXED_NHEAD=8` (`scripts/optimize.py`, `shape` target only) — passed
to `nn.objective` instead of being searched, removing 2 of the (now 13) dimensions.

## Evidence

Independently reproduced from `results/shape_v10/shape/lead_{6,12,24}h/best_trial.txt`,
`results/shape_v11/shape/lead_{6,12}h/best_trial.txt`, and
`results/shape_v12/shape/lead_{12,24,48}h/current_best.txt` — 8 (study, lead_time) data
points:

| study, lead | head_dim | nhead |
|---|---|---|
| v10, 6h | 32 | 8 |
| v10, 12h | 32 | 8 |
| v10, 24h | 32 | 4 |
| v11, 6h | 32 | 4 |
| v11, 12h | 32 | 8 |
| v12, 12h | 8 | 8 |
| v12, 24h | 16 | 8 |
| v12, 48h | 32 | 8 |

`head_dim=32` wins 6/8; `nhead=8` wins 6/8 — matching the code comment's own tally exactly.
No other hyperparameter showed comparable cross-lead-time/cross-version agreement:
`seq_len`/`batch_size`/dropouts/`weight_decay` each span nearly their entire search range
across these same 8 points with no consensus value, and `num_encoder_layers`' superficially
similar 5/8-for-value-4 split included `shape_v10`'s 6h run picking a very different value
(1) — suggesting a real lead-time-dependent capacity need there, not noise, unlike
`head_dim`/`nhead`.

## Decision

Kept. This is a genuine data-driven simplification of the search space, not a hypothesis
awaiting further testing — the 6/8 agreement across three separate studies (different
`STUDY_VERSION`s, different loss regimes pre/post [[016]]/[[020]]) is the strongest
cross-study convergence signal found among any hyperparameter in this project's Optuna
history.

## Related

Code: `scripts/optimize.py` (`FIXED_HEAD_DIM`, `FIXED_NHEAD`), `nn/optimization.py::
objective` (`fixed_head_dim`/`fixed_nhead` parameters)
Other decisions: [[026]] (the search-space expansion this offsets)
Manuscript section this might feed: Methods — hyperparameter search (final architecture
choice for the `shape` target).
