---
status: kept
date: 2026-08-21
commits: [96c704e, 197885c, a89b19c, 46fa70f, c2cd33a]
category: training
---

# 026 — Composite KL + Wasserstein + Peak loss ablation: `combined` recipe adopted over the plain per-bin loss

## Context

`scripts/ablate_loss.py` ran a small, fixed-architecture Optuna study (`STUDY_VERSION
lossablation_v2`, architecture pinned from `shape_v12`'s `lead_12h` reference config) to
test whether substituting the plain per-bin loss (RMSE/MSE) with some combination of
`SpectralKLDivergenceLoss`, `SpectralWassersteinLoss` (W2, see [[024]]), and
`SoftPeakHeightLoss` improves peak/multimodal fidelity — the recurring finding motivating
[[020]] and related entries is that aggregate spectrum-wide error metrics (Shape_RMSE/SS)
dilute peak-specific behaviour, so this ablation judged each phase by a wind-sea/swell-
conditioned panel (`Peak_Height_RelError`, `Peak_Separation_Recall`, `Tm02_RMSE`,
`Tm02_Bias`, each split by partition label via [[008]]'s classifier) instead.

## Change

Seven phases were run: `baseline` (per-bin loss only, 5x for run-to-run variance),
`kl`, `wasserstein_only`, `peak_only`, `wasserstein` (KL-weight fixed from the `kl`
phase's winner), `peak` (same), and `combined` (KL fixed, Wasserstein+Peak weights jointly
searched). `base_loss_weight=0` for every non-`baseline` phase — a literal substitute of
the per-bin loss, not an addition to it, matching the original proposal `L = D_KL +
lambda_1·W2 + lambda_2·L_peak`.

## Evidence

`results/lossablation_comparison_v2.md` (test-set numbers, target=shape, lead=12h,
reproduced/verified directly, not merely transcribed from a comment):

| phase | Peak_Height_RelError (avg) | Peak_Separation_Recall (avg) | Tm02_RMSE (avg) | Tm02_Bias (avg) |
|---|---|---|---|---|
| baseline | 0.4141 | 0.6628 | 0.4858 | −0.1583 |
| combined | 0.3840 | 0.9425 | 0.4126 | 0.0201 |

`combined` beats `baseline` on all four averaged metrics. At the wind-sea/swell sub-split
level, `combined` beats `baseline` on 7 of 8 splits; the one exception is `Tm02_Bias`
(wind-sea): baseline 0.0054 vs. combined 0.0074 (both very close to the ideal 0 — combined
is marginally worse there, not a meaningful regression). No other single-term phase (`kl`
alone, `wasserstein`/`wasserstein_only` alone, `peak`/`peak_only` alone) matched this: each
traded away at least one of the four metric families relative to baseline (e.g.
`peak_only` and `peak` show sharply worse `Peak_Height_RelError`/`Tm02_RMSE`/`Tm02_Bias`
despite the best raw `Peak_Separation_Recall`).

Winning weights (test): `kl_loss_weight=45.87`, `combined` phase:
`wasserstein_loss_weight=76.56, peak_loss_weight=8.802`.

This is a controlled, single-architecture comparison (architecture pinned, only loss
composition varies per phase) — not confounded by architecture search, unlike several
earlier entries in this log.

## Decision

Kept. `combined` (`base_loss_weight=0`, KL + Wasserstein-2 + Peak) was promoted from
ablation-only into the real Optuna search space in `scripts/optimize.py` v13 (see
[[027]]), with `OBJECTIVE_METRIC` switched to `'peak_fidelity_SS'` accordingly (see
[[009]]) since training no longer optimizes RMSE at all under this recipe.

## Related

Code: `scripts/ablate_loss.py`, `nn/training_loop.py::train_one_epoch` (`base_loss_weight`),
`utils/loss.py`
Other decisions: [[008]] (wind-sea/swell partition classification used for the scoreboard),
[[009]] (`peak_fidelity_SS`, the objective metric this result motivated for v13), [[020]],
[[024]] (the individual loss terms combined here), [[027]] (promotion into the real search
space)
Manuscript section this might feed: Results — loss-function ablation (this is a genuine
reported ablation, not project-history narration — see parent CLAUDE.md §6's distinction
between the two).
