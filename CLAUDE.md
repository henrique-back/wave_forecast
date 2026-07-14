# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project applies transformer-based deep learning to wave spectra forecasting. A buoy produces 4-channel directional wave spectra (spectral density, alpha1, alpha2, r1) over a log-spaced frequency grid (47 bins, 0.02–0.485 Hz). The model forecasts one of three targets, for lead times of 6/12/24/48 hours:
- **`hs`**: scalar significant wave height Hs = 4√m₀
- **`density`**: the full physical spectrum E(f)
- **`shape`**: the unit-area normalized spectrum E(f)/m₀ — see "Shape/magnitude model split" below

Hyperparameter search is managed by Optuna with a SQLite backend (`optuna_study_{STUDY_VERSION}.db`).

## Commands

All commands should be run from the repo root with the virtualenv active:

```bash
source .venv/bin/activate
```

**Run tests:**
```bash
pytest tests/
```

**Run a single test:**
```bash
pytest tests/test_spectral.py::TestJONSWAPRoundTrip::test_hs_after_roundtrip
```

**Preprocess buoy data** (only needed once per buoy; output cached as `buoy_data/{BUOY_ID}/processed_data.pkl`):
```bash
python scripts/data_processing.py
```

**Run hyperparameter optimization** (edit the config constants at the top of the file first — `EXPERIMENT_NAME`, `target`, `CHANNEL_SET`, `AUX_SET`, `OBJECTIVE_METRIC`, `lead_times_hours`):
```bash
python scripts/optimize.py
```

**Retrain a final model** from an Optuna study's best hyperparameters and evaluate on the held-out test set:
```bash
python scripts/train.py
```

**Inspect a trained model's predictions** on test-set samples (autoregressive inference + plots):
```bash
python scripts/infer.py --experiment <EXPERIMENT_NAME> --lead 6
python scripts/infer.py --experiment hs_shape_v5 --target combined --lead 6
```

**Compare forecast accuracy across two or more experiments** on the full physical density spectrum, regardless of whether each was trained as a monolithic `density` model or a `hs`+`shape` pair recombined at inference time:
```bash
python scripts/compare_versions.py \
    --experiment weightedmeanSS_conv_freqemb_v3:density \
    --experiment hs_shape_v5:combined:"HS+Shape v5" \
    --lead 6
```

**Regenerate `results/RESEARCH_LOG.md`** (a cross-experiment summary table; also run automatically at the end of `optimize.py`):
```bash
python scripts/summarize_results.py
```

**Plot results** (reads `results/` directory):
```bash
python scripts/plot_results.py
```

## Architecture

### Data pipeline (`utils/`, `nn/prepare_x.py`, `nn/prepare_aux.py`, `nn/prepare_y.py`)

Raw buoy `.txt` files → `utils/data_processing.py` reads/reindexes/interpolates them → saved to `buoy_data/{BUOY_ID}/processed_data.pkl`. Returns a 5-tuple `(density, alpha_1, alpha_2, r_1, wind)`. The four spectral DataFrames (`density`, `alpha_1`, `alpha_2`, `r_1`) share a full hourly datetime index with shape `(time, num_freqs)`. `wind` has columns `['wind_u', 'wind_v']` (same hourly index, no freq axis) — derived from `buoy_data/wind.txt` (NDBC stdmet format) by `process_wind()`: NDBC sentinels (`WDIR==999`, `WSPD==99.0`) are masked to NaN, then converted to u/v components (`u = -WSPD·sin(WDIR)`, `v = -WSPD·cos(WDIR)`) **before** time-interpolation, since WDIR is circular and interpolating the raw angle across a gap would pass through the wrong side of the compass.

Data is split **70 % train / 15 % val / 15 % test** (chronological, no shuffling across splits), computed identically inside `nn/optimization.py::_prepare_dataloaders` for both the Optuna search and `scripts/train.py`'s final retrain so their metrics stay comparable.

**Input channels are configurable along two independent axes** (`nn/channels.py`), both threaded through `nn/optimization.py::_prepare_dataloaders`/`objective()` and set in `scripts/optimize.py`/`scripts/train.py`:
- `channel_set` — which frequency-resolved channels are stacked by `prepare_X` into `(samples, seq_len, num_freqs, num_channels)`: `'density'` (density only) or `'full'` (density + alpha_1 + alpha_2 + r_1, the original 4-channel input). `density` is always first — `utils/get_start_token.py` reads channel 0 as the decoder start token.
- `aux_set` — which scalar-per-timestep side channels are fused into the encoder separately (not frequency-resolved, so they bypass `prepare_X`/`FreqDimEmbedding` entirely): `'none'` or `'wind'` (`wind_u`, `wind_v`, windowed by `nn/prepare_aux.py` into `(samples, seq_len, num_aux_channels)`). Wind is encoder-only context — it never reaches the decoder, since it's never a forecast target.

**Normalization** (inside `nn/optimization.py::_prepare_dataloaders`, modes registered in `nn/channels.py`):
- `density`: scale-only (`E / μ(f)` per-frequency mean). Non-negativity must be preserved because `compute_hs` calls `sqrt(trapz(E))`. Always normalized regardless of `channel_set`, since it also produces the `density`-target `y`.
- `alpha_1`, `alpha_2`, `r_1`, `wind_u`, `wind_v`: z-score.

`freq_means` — the per-frequency training mean μ(f) — is the single tensor that bridges normalized and physical space. It flows through training, evaluation, inference, and the start token. For the `hs`/`shape` targets, `prepare_y` builds `y` from the **physical** (pre-normalization) density so the target and persistence baseline are meaningful without needing `freq_means` at loss time; for `density` targets, `y` stays in normalized space and `freq_means` is applied externally at loss/metric time.

### Shape/magnitude model split

Rather than always training one `density`-target model, an alternative is to train **two separate models** at the same lead time — one `hs`-target model (predicts scalar magnitude) and one `shape`-target model (predicts the unit-area spectrum E(f)/m₀, via `compute_shape` in `utils/compute_hs.py`) — and recombine their outputs at inference time:

```
E_pred(f, t) = shape_pred(f, t) * m0_pred(t),   m0_pred = (Hs_pred / 4)^2
```

This decouples the (typically easier) magnitude-forecasting problem from the (typically harder) shape-forecasting problem, motivated by a single `density`-target model tending to underestimate spectral peaks and over-smooth the high-frequency tail. `scripts/infer.py --target combined` loads a matching `hs` checkpoint and `shape` checkpoint (same experiment/lead) and does this recombination for inspection; `scripts/compare_versions.py --experiment name:combined` does the same across the full test set for metric comparison against monolithic `density` models. Example experiments using this split: `hs_shape_v5`, `hs_shape_v6`.

### Model (`nn/transformer.py`)

`WaveHeightBaselineNN` is an encoder-decoder Transformer (`nn.Transformer`) with two custom front-ends:
- **Encoder embedding** (`nn/freq_embedding.py::FreqDimEmbedding`) — a shared linear maps the multi-channel measurement at each frequency bin into a per-bin representation, then all bins are aggregated into a single `embed_dim` token. This gives the model a structural prior that channels at the same frequency are related, instead of flattening `(num_freqs, num_channels)` through one large `Linear` (`nn/embedding.py::Embedding`, still used for the decoder's scalar/flat input).
- **Temporal conv front-end** (`nn/temporal_conv.py::TemporalConvFrontend`, encoder only) — three dilated `Conv1d` layers (dilation 1, 2, 4) with pre-norm residuals extract local temporal patterns at 3h/5h/9h scales before the global self-attention layers. Not applied to the decoder, which is short (≤ lead_time steps) and already uses a causal mask.

When `num_aux_channels > 0`, the auxiliary side-input (e.g. wind) is embedded separately via `self.aux_embedding` and added additively into the per-timestep encoder token before positional encoding, since aux channels aren't frequency-resolved. The decoder receives either a scalar Hs start token, a unit-area shape start token, or the last observed normalized spectrum, depending on `target`. Autoregressive inference is in `infer()`, which encodes `src` once and reuses the cached `memory` across all decode steps.

Embed dim is always `head_dim × nhead` (enforced in `optimization.py`) to guarantee divisibility.

`nn/lstm.py` contains an older encoder-only LSTM baseline (same class name, different interface — no `infer()`, single-step output). It is not wired into the current training/optimization pipeline.

### Training (`nn/training_loop.py`, `nn/optimization.py`)

Training uses teacher forcing with **scheduled sampling**: `tf_ratio` decays linearly from 1.0 → 0.0 over `4 × patience` epochs, closing the teacher-forced/autoregressive gap. For `density` targets, the RMSE loss is computed in **physical space** (denormalized via `freq_means`) so the gradient is not dominated by normalization artifacts.

`nn/optimization.py::_train_model` is shared by `objective()` (Optuna trial) and `scripts/train.py` (fixed-config final retrain) so the two never drift apart — the only behavioral difference is that a `trial` object, when passed, reports per-epoch scores to Optuna and can prune the run.

Early stopping patience=20 (Optuna trials) / configurable in `scripts/train.py`; LR scheduler `ReduceLROnPlateau(patience=5, factor=0.5, cooldown=2)` fires before early stopping. Gradients are clipped to `max_norm=1.0` per step.

### Evaluation metrics (`nn/evaluate.py`)

The persistence baseline is the last observed value (start token) broadcast over all lead-time steps. Skill Score `SS = 1 − RMSE_model / RMSE_persistence` is the primary optimization target family.

`evaluate()` always returns `Hs_SS` (for `target == 'hs'` this equals `overall_SS` exactly; for `density` it's derived from denormalized spectra) — robust to `seq_len` variation, since the persistence baseline's RMSE changes with `seq_len` but Hs itself does not.

For `density` target with `freq_means`, the evaluator additionally computes bulk parameters: `Hs_RMSE`/`Hs_Bias`, `Tm02_RMSE`/`Tm02_Bias`, `Shape_RMSE` (spectral shape error decoupled from energy magnitude, masked below `M0_MASK_THRESHOLD`), and `SI_per_bin`/`SI_mean` (per-frequency Scatter Index). All integrations use `np.trapezoid` on the physical (denormalized) spectra.

### Optuna study (`scripts/optimize.py`)

Study name pattern: `{target}_{channel_set}_{aux_set}_{lead_Nh}_{STUDY_VERSION}` (e.g. `hs_full_wind_lead_24h_v7`) — `channel_set`/`aux_set` are included so a study never silently mixes trials across incompatible input configs. Results are saved under `results/{EXPERIMENT_NAME}/{target}/lead_{N}h/` (channel_set/aux_set/architecture are recorded in that experiment's `metadata.md`, written once by `optimize.py` and consumed by `summarize_results.py` — follow the existing convention of giving a new `EXPERIMENT_NAME` when input variables or architecture change). Bump `STUDY_VERSION` when the objective function or hyperparameter space changes incompatibly; the old DB file is kept at its old version and a fresh one is created.

Sampler: `TPESampler(n_startup_trials=20, multivariate=True, seed=42)`. Pruner: `MedianPruner(n_warmup_steps=30, n_min_trials=5, interval_steps=1)` — the 30-step warmup avoids over-pruning right at the boundary (54% of 12h trials were pruned at exactly epoch 20 with `n_warmup_steps=20` in an earlier study). After each trial, `save_progress` writes a checkpoint to the results folder, and a trial hitting `torch.OutOfMemoryError` is caught and marked failed rather than aborting the whole study.

`OBJECTIVE_METRIC` is set per run at the top of `optimize.py`. Available values: `weighted_mean_SS` (recommended default — exponentially-weighted mean per-step Skill Score, robust to variable `seq_len`), `overall_SS`, `Hs_SS`, `RMSE`, `Hs_RMSE`, `Tm02_RMSE`, `Shape_RMSE`, `SI_mean` (last four are `density`-target only).

At the end of a run, `optimize.py` automatically invokes `scripts/summarize_results.py` to regenerate `results/RESEARCH_LOG.md`.

### Note on `deltat` / older results layout

Some utility and inference code (`scripts/infer.py`, `scripts/plot_results.py`, `scripts/summarize_results.py`) still references a `deltat` (downsampling stride) folder level, e.g. `results/{EXPERIMENT_NAME}/{target}/deltat_{N}/lead_{N}h/`. The current `scripts/optimize.py`/`scripts/train.py` no longer thread a `deltat` parameter through `_prepare_dataloaders` and write results directly under `results/{EXPERIMENT_NAME}/{target}/lead_{N}h/` (no `deltat_{N}` level) — e.g. `wind_combined_v7`, `hs_shape_v5`, `hs_shape_v6`. `scripts/infer.py::find_checkpoint` checks the `deltat_{N}`-nested path first and falls back to the flat path, so it works with both conventions; be aware of this when adding new tooling that walks the `results/` tree.

## Key Invariants

- **Always denormalize before physical integrations.** `E_phys = E_norm * freq_means`. Passing `E_norm` directly to `compute_hs` / `compute_bulk_params` / `compute_shape` produces wrong results — this is explicitly tested in `tests/test_spectral.py`.
- **`freq_means` is fit on training split only**, then applied to val/test splits.
- **Do not call `set_seed()` inside `objective()`** — it collapses per-trial variance that Optuna needs for TPE.
- **Skill Score is not equivalent to RMSE across Optuna trials** because `seq_len` is a hyperparameter, so the persistence baseline varies between trials. Prefer `Hs_SS` or `weighted_mean_SS` over raw `RMSE` for this reason.
