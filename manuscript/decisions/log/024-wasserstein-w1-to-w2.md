---
status: kept
date: 2026-08-17
commits: [62e9970]
category: training
---

# 024 — `SpectralWassersteinLoss`: Wasserstein-1 → Wasserstein-2

## Context

[[020]] introduced `SpectralWassersteinLoss` using the 1-D Wasserstein-1 (earth-mover)
distance, computed via its convenient exact shortcut for p=1 — the L1 distance between
CDFs, `∫|CDF_pred(f)−CDF_true(f)| df`. W1's linear transport cost treats "one peak moved
far" and "many small local shifts summing to the same total distance" as interchangeable,
which does not match the failure mode motivating this loss: a peak fully misplaced to a
distant frequency should cost disproportionately more than the same peak merely broadening
into its immediate neighbourhood.

## Change

Switched `SpectralWassersteinLoss` to Wasserstein-2 (quadratic transport cost). W1's
CDF-L1 shortcut is specific to p=1 and has no p=2 analogue, so W2 required implementing the
general 1-D formula in the quantile (inverse-CDF) domain instead:
`W_p(F,G)^p = ∫_0^1 |F^{-1}(q) − G^{-1}(q)|^p dq`. Since the true spectrum's own CDF
trivially inverts (its quantile function at its own CDF values is just `freqs` itself),
only the predicted spectrum's quantile function needs inverting (`utils/loss.py::
_inverse_cdf`, via `torch.searchsorted` + linear interpolation, differentiable), evaluated
at the true spectrum's own quantile levels — this doubles as the numerical quadrature nodes
for the integral, avoiding a variable per-sample breakpoint count and keeping every tensor
batchable.

## Evidence

Reasoning-only (mathematical/methodological argument, not an isolated before/after metrics
comparison) — implemented as part of the same day's work that also added
`_peak_windows_for_batch` ([[025]]). The subsequent loss-ablation study ([[026]]) validated
the *combined* recipe under W2, not W1 vs. W2 in isolation.

**Important downstream consequence, disclosed at the time in code**: weights tuned under
W1 (e.g. [[020]]'s manually-swept 50/150 sweep, and `shape_v12`'s own
`wasserstein_loss_weight=181.67` at lead_12h) are on a different numeric scale than W2's —
W1's `∫|CDF gap| df` and W2's `sqrt(∫(quantile gap)² dq)` are different quantities, not a
rescaling of each other.

## Decision

Kept. **Flag for the manuscript**: `manuscript/decisions/wasserstein_kl_justification.tex`
— the standalone LaTeX document arguing for this loss term's inclusion — currently argues
entirely in terms of Wasserstein-1 and does not mention this switch. Any Methods/Discussion
prose drawn from that document needs to be checked against the current W2 implementation
before being used, and the `.tex` document itself likely needs updating to argue for W2
(or explicitly justify W1 as a design choice that was later revised) before it's treated as
the source for that paragraph. This was not fixed as part of writing this entry — flagged
per the parent CLAUDE.md §9's instruction to surface such issues rather than resolve them
silently.

## Related

Code: `utils/loss.py::SpectralWassersteinLoss`, `utils/loss.py::_inverse_cdf`
Other decisions: [[020]] (original W1 term, whose tuned weights this invalidates), [[026]]
(loss-ablation study conducted under W2)
Manuscript section this might feed: `manuscript/decisions/wasserstein_kl_justification.tex`
needs revision — see Decision above.
