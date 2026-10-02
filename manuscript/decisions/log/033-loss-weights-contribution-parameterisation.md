---
status: kept
date: 2026-10-02
commits: []
category: training
---

# 033 — Composite-loss weights sampled as contributions, not raw multipliers

## Context

[[032]] closed with an open item: `nn/optimization.py::objective` sampled
`kl_loss_weight ∈ [0.1, 50]`, `wasserstein_loss_weight ∈ [1, 200]` and
`peak_loss_weight ∈ [0.01, 20]` independently with `base_loss_weight = 0`, so a heavily
peak-dominated draw could re-enter the degenerate regime [[032]] documents. It was recorded as an
edge-case exposure.

Measuring the terms shows that was an understatement. `nn/training_loop.py::train_one_epoch` sums
four terms **raw**, and they are expressed in four unrelated units:

| term | median raw magnitude | units |
|---|---|---|
| base (trapz-weighted MSE) | 6.84 | log-space squared error |
| W2 | 0.0085 | Hz |
| KL | 0.064 | nats |
| peak | 196.7 | E² (shape) |

(`scripts/measure_loss_term_scales.py`, `lossablation_v3` `kl`-phase checkpoint, buoy 32012,
target `shape`, lead 12 h, post-[[030]] detector, 6 training batches under teacher forcing.)

A weight's numeric value therefore says nothing about how much its term contributes. Converting
v13's bounds through these magnitudes, over 20 000 log-uniform draws:

| | median KL share | median W2 share | median peak share | P(peak > 90% of loss) | median peak:KL | 99th pct peak:KL |
|---|---|---|---|---|---|---|
| **v13** | 0.002 | 0.001 | **0.995** | **87.2%** | 616 | 222 071 |
| **v14** | 0.245 | 0.202 | 0.207 | 5.0% | 0.63 | 18.8 |

So v13's search was not *exposed* to the peak-dominated regime — it was **centred in it**. The
median draw gave the peak term 99.5% of the loss with KL and W2 inert, and 87% of the space was
>90% peak. Across 80 trials × 3 lead times the search was effectively one-dimensional.

Two things kept this invisible: `train_one_epoch` returned only `{'RMSE': avg_loss}`, the sum of all
four terms, with no breakdown; and `utils/loss.py::SoftPeakHeightLoss`'s docstring had asked for
exactly that breakdown — *"if this is ever wired into training, log it as its own component
(mirroring DirectionalLoss's `components` return)"* — but it was never done after the term was
wired in.

## Change

**Per-term logging** (the half that makes the rest observable):
- `nn/training_loop.py::train_one_epoch` now returns
  `{'RMSE': avg_loss, 'loss_components': {'base','w2','kl','peak'}}` — each term's mean per-sample
  **weighted** contribution, summing to `avg_loss`. A zero-weight or skipped term records `0.0`, not
  NaN. Follows `DirectionalLoss`'s `(total, components)` convention (`utils/loss.py:438-481`).
- `nn/optimization.py::_train_model` prints the shares in the per-epoch line
  (`loss mix: peak 92% kl 8%`) whenever more than one term is active, so a dominated run is visible
  at epoch 1 rather than after a study.
- `_train_model` carries the selected epoch's shares out via `best_val_metrics` as
  `train_loss_share_*`; `objective` and `scripts/ablate_loss.py`'s own objective record them in
  `trial.user_attrs`.

**Contribution parameterisation:**
- New `nn/optimization.py::LOSS_TERM_REFERENCE = {'base': 6.84, 'kl': 0.064, 'w2': 0.0085,
  'peak': 197.0}` — the measured medians above.
- `objective` samples a scale and two ratios **anchored on KL**, replacing the three raw-weight
  draws:
  ```
  kl_contrib ~ logU(0.05, 5.0)      w2_rel ~ logU(0.02, 20.0)      peak_rel ~ logU(0.02, 20.0)
  weight_i = kl_contrib * rel_i / LOSS_TERM_REFERENCE[i]
  ```
  KL is the anchor because with `base_loss_weight=0` it is the only full-support,
  position-determining term left, so expressing the others relative to it is what bounds the drift
  away from a position-determined loss. Worst-case imbalance falls from ~6×10⁵:1 to 20:1.
- **Calibration:** `lossablation_v3`'s `peak` arm (`kl=19.86`, `peak=0.0746`) realises a peak:KL
  contribution ratio of **11.7** and is well behaved, while `peak_only` (ratio → ∞, KL absent)
  collapses. The ceiling of 20 therefore sits above the known-good point rather than below it. The
  0.02 floor lets TPE discover a term isn't needed without reaching exactly zero — which for the
  peak term is the degenerate direction.
- `kl_contrib` is the honest name for what KL's weight already was: with no base term to balance
  against, it sets the loss's overall scale, as `scripts/ablate_loss.py`'s `kl`-phase comment notes.

**Backwards compatibility** — `nn/optimization.py::resolve_loss_weights(params)` maps either
parameterisation to the three absolute weights (v14+ `kl_contrib`/`w2_rel`/`peak_rel`, or pre-v14
absolute weights passed through). `scripts/train.py` reads through it instead of
`params.get(key, 0.0)`, so a v13 study still retrains and the derivation has one definition. It
raises on a params dict carrying neither, rather than silently zeroing — the old `.get` default
turned a renamed parameter into an all-zero loss that surfaced as the misleading *"predates the
KL/Wasserstein/Peak loss (v13)"* error.

**`scripts/optimize.py` → v14**: `STUDY_VERSION = "v14"`, `EXPERIMENT_NAME = "shape_v14"`, and
`FIXED_HEAD_DIM = FIXED_NHEAD = None` — [[031]]'s other open item, since [[027]] pinned 32/8 on a
tally over studies selected by the pre-[[030]] detector and pre-[[031]] criterion, both superseded.
13 tunable hyperparameters again.

**`scripts/measure_loss_term_scales.py`** regenerates `LOSS_TERM_REFERENCE` and flags any term that
has drifted more than 2× from the committed value. CPU-only and read-only, so no Slurm gate.

## Evidence

- Measurement reproduces the committed constants to within 1% (`base` 1.00×, `w2` 1.00×, `kl` 0.99×,
  `peak` 1.00×).
- The 20 000-draw comparison in the table above, and `tests/test_loss_weight_parameterisation.py`,
  which pins both the new space's shape (`peak_rel` median O(1), bounded [0.02, 20], the 11.7
  known-good point representable) and the diagnosis itself (recomputing v13's bounds through the
  references still yields >100:1 median and >1×10⁵ worst case).
- `resolve_loss_weights` verified against the real
  `results/shape_v13/shape/lead_12h/best_trial.txt`: weights pass through unchanged and the retrain
  guard does not fire.
- 173 tests pass.

## Decision

- **Kept.** Loss weights are sampled as contributions relative to measured term magnitudes, and
  every run records its realised loss composition.
- **Resolves** [[032]]'s "still open" production-search item — and corrects its framing: the search
  was centred in the imbalanced regime, not merely exposed to it at the edge.
- **Superseded:** v13's `trial.params` loss weights are not comparable to v14's. v13's results stay
  on disk; its studies remain retrainable through `resolve_loss_weights`.
- **Not claimed:** that a balanced composition performs better. `lossablation_v3` so far has
  `baseline` 0.4572, `kl` 0.4567, `wasserstein` 0.4545, `peak` 0.4640 — all inside the `baseline`
  replicate spread of 0.035 — so the composite loss may not be earning its place at all. This entry
  makes the search space interpretable; whether to keep a multi-term loss is the ablation's question,
  pending `combined`.
- **Still open:** `LOSS_TERM_REFERENCE` is a fixed constant and will drift if the detector, target,
  normalisation or grid changes. `measure_loss_term_scales.py` detects that but nothing enforces
  re-running it — a CI check, or asserting at study start that the references still hold, would close
  it. The v14 search is **not launched**; that waits on the ablation.

## Related

- Code: `nn/optimization.py` (`LOSS_TERM_REFERENCE`, `resolve_loss_weights`, `objective`,
  `_train_model`); `nn/training_loop.py::train_one_epoch`; `scripts/train.py`;
  `scripts/optimize.py`; `scripts/ablate_loss.py`; `scripts/measure_loss_term_scales.py`;
  `tests/test_loss_weight_parameterisation.py`.
- Other decisions: [[024]] (W1→W2, which changed what the W2 weight balances against), [[026]] (the
  ablation being re-run), [[027]] (the head_dim/nhead pin this lifts), [[030]] (the detector change
  that rescaled L_peak), [[031]] (`peak_fidelity`), [[032]] (why the peak term cannot stand alone).
- Manuscript: the methods' loss description should state the contribution parameterisation and cite
  the reference magnitudes; no v13 loss weight should be quoted as comparable to a v14 one.
