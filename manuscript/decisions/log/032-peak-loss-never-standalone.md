---
status: kept
date: 2026-10-02
commits: []
category: training
---

# 032 — `SoftPeakHeightLoss` is never an objective on its own; the `peak_only` arm is removed

## Context

The `lossablation_v3` re-run ([[031]], [[030]]) included a `peak_only` arm: `base_loss_weight=0`,
`kl_loss_weight=0`, `wasserstein_loss_weight=0`, searching `peak_loss_weight` alone. It asked
`wasserstein_only`'s question for the peak term — does it need KL underneath it, or does it work
alone?

It scored `peak_fidelity` 0.0023 against 0.4387 ± 0.0137 for the `baseline` replicates. That is not
a weak result, it is a structural one, and the reason matters enough to record so the arm is not
re-added.

## Why the term cannot be an objective by itself

For each true peak window *k*, with `H = Σ_i E_i σ_i`, `σ = softmax(E_pred/τ_k)`,
`τ_k = c·H_true·Δf_k/freq_span`:

```
loss = mean_k (H_pred − H_true)²
∂L/∂E_j = 2(H_pred − H_true) · σ_j · [1 + (E_j − H_pred)/τ]
```

Three consequences, all by design:

1. **One scalar per window.** Measured on the test split: 2.51 windows per sample. The objective
   therefore imposes ~2.5 scalar constraints on a 47-dimensional output, leaving ~44 degrees of
   freedom untouched. (Window *coverage* is 103% — the [[030]] combining makes them tile — but
   coverage is irrelevant: each window contributes a single number, not a per-bin penalty.)
2. **Support confined to the prediction's own maxima.** σ is sharply peaked at τ ≈ 0.026·H_true, so
   any bin more than a few τ below the window's current max receives no gradient at all. The term
   is silent — not weakly opinionated — about flanks, valleys and the tail.
3. **Translation-invariant.** The class docstring states it: *"by construction invariant to pure
   translation… positionless"*. `H_true` enters only as a target height, never as a location.

Together these mean the minimiser is a **manifold**, not a point: any spectrum whose soft-max height
inside each window matches `H_true` is an exact optimum, wherever those maxima sit. It is a proper
scoring rule for ~2.5 functionals of the spectrum, not for the spectrum. The term is a sharpening
operator applied at a self-nominated location — useful once a position-determining term has put the
prediction's peak roughly in the right place, meaningless before that.

This is what `utils/loss.py::SoftPeakHeightLoss`'s own docstring already said: the terms *"are meant
to be summed, each covering the other's blind spot, **not to replace one another**."* The
`peak_only` arm tested a configuration the loss was documented never to support.

## Evidence

**The optimiser succeeded; the objective did not identify the target.** `lossablation_v3`,
`peak_only` trial 0 (`peak_loss_weight = 0.0746`): training loss 5.91 → 3.09 → 2.73 → … → 1.90 by
epoch 21, descending normally, while validation CC stayed at **0.03–0.08** and Shape_SS at −2.1 to
−3.0. A loss that falls while correlation with the truth stays at zero is the signature of an
objective satisfiable without learning the task.

**Not a weight-tuning problem.** Trial 0 used 0.0746 and trial 1 used 56.70 — nearly three orders of
magnitude apart — scoring 0.0023 and 0.0005.

**An accidental controlled comparison.** `peak` and `peak_only` both drew
`peak_loss_weight = 0.0745934328572655` (identical: same `TPESampler(seed=42)`, same range, same
trial index). The only difference is `kl_loss_weight`:

| phase | `kl_loss_weight` | val `peak_fidelity` | val CC | val Shape_SS |
|---|---|---|---|---|
| `peak` | 19.86 | 0.4640 | 0.73 → 0.83 | −0.79 → −0.08 |
| `peak_only` | 0 | 0.0023 | 0.03 → 0.08 | −2.1 → −3.0 |

Identical peak term and weight; adding a full-support, position-determining term is the entire
difference.

**Scored on its own loss, it is the worst arm.** Final-step autoregressive test set,
`SoftPeakHeightLoss` against true-derived windows: `peak_only` 645.2, versus `kl` 40.3, `peak` 43.1,
`wasserstein_only` 50.8, `wasserstein` 51.7, `baseline` 59.6 — and `baseline` never saw the peak
term. Mean predicted/true peak-height ratio 3.30 for `peak_only` against 0.94–1.11 for every other
arm.

**The dividing line is predictive.** Sorting the v3 arms by whether their loss is a proper,
full-support function of the whole output vector sorts the results exactly: per-bin MSE
(`baseline` 0.457), forward KL (`kl` 0.457) and W2 (`wasserstein_only` 0.399) each determine a
spectrum and each stands alone; `SoftPeakHeightLoss` does not and does not (0.002). `wasserstein_only`
is the control that matters — a single-term substitute arm is *not* doomed in general, only this one.

## Caveat on the reported checkpoint

`peak_only`'s saved checkpoint is from **epoch 3** (`val_score = 0.002318`, the only epoch whose
smoothed `peak_fidelity` exceeded zero; early stopping then ran out the patience at ~epoch 23). So
its plotted spectra in `results/lossablation_spectra_v3*.png` — a near-fixed comb, across-sample
coefficient of variation 0.28 against the truth's 1.13 — reflect an essentially untrained network as
much as any attractor of the loss. The conclusion does not rest on those plots: it rests on the
training trajectory, the controlled comparison above, and the structural argument. A side lesson:
selecting the best epoch on a metric that is identically 0.0000 freezes an arbitrary early epoch.

## Change

- `scripts/ablate_loss.py`: `peak_only` removed from `PHASE_N_TRIALS`/`PHASE_N_STARTUP` (the single
  source of the phase list for `compare_ablation_phases.py`, `evaluate_ablation_phases.py` and
  `plot_ablation_spectra.py`, so it disappears everywhere at once) and from the `--phase` choices.
  `_fixed_weights_for_phase("peak_only")` now raises `ValueError` with the reason rather than falling
  through to the generic unknown-phase error, so a stale `--phase peak_only` cannot resurrect it
  quietly.
- `slurm/ablate_peak_only.slurm` deleted; `slurm/submit_ablation.sh` down to 6 phases.
- The running Slurm job for this phase (91, requeued after a `NODE_FAIL`) was cancelled.
- `tests/test_ablate_loss.py`: the arm's behaviour test replaced by one asserting it is refused, plus
  one asserting it is absent from the phase list.
- `peak`'s `peak_loss_weight` upper bound of 100 is annotated as safe *only because* that phase fixes
  `kl_loss_weight` nonzero — letting the peak term dominate the gradient walks back toward this
  regime.

v3's existing `results/lossablation_peak_only_lossablation_v3/` and its Optuna study are left in
place as the evidence for this entry.

## Decision

- **Kept.** `SoftPeakHeightLoss` is only ever a term alongside a full-support, position-determining
  loss. Do not re-add a standalone arm.
- **Not affected:** `peak` (KL + peak) and `combined` keep the term; this entry says nothing about
  whether it earns its place *there*. On current v3 numbers it may not — `kl` 0.4567 vs `peak`
  0.4640, a gap of 0.007 against a baseline spread of 0.035 — but that is the ablation's open
  question, not this one.
- **Resolved by [[033]]** (which found the exposure far larger than described here — the v13
  search space's *median* draw gave the peak term 99.5% of the loss): a related exposure in the
  production search. `nn/optimization.py::objective` draws
  `kl_loss_weight ∈ [0.1, 50]` and `peak_loss_weight ∈ [0.01, 20]` independently with
  `base_loss_weight=0`, so while a literal peak-only draw is impossible (KL's floor is 0.1), a
  `kl≈0.1, peak≈20` draw is heavily peak-dominated — the same regime by degree. Worth a floor on the
  KL:peak ratio, or a reparameterisation into (scale, mixing weights), when v14's search space is
  settled.

## Related

- Code: `utils/loss.py::SoftPeakHeightLoss`; `nn/training_loop.py::_peak_windows_for_batch`,
  `train_one_epoch`; `scripts/ablate_loss.py`; `slurm/submit_ablation.sh`.
- Other decisions: [[025]] (per-batch windows), [[026]] (the ablation being re-run), [[028]] (window
  displacement check), [[030]] (detector fix — widened windows raise τ and so the term's floor),
  [[031]] (the `peak_fidelity` metric these runs are scored on).
- Manuscript: the loss-ablation results section should report `peak_only`'s removal as a structural
  finding about the term, not as a missing arm.
