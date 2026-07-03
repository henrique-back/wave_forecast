# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project applies transformer-based deep learning to wave spectra forecasting. A buoy produces 4-channel directional wave spectra (spectral density, alpha1, alpha2, r1) over a log-spaced frequency grid (47 bins, 0.02–0.485 Hz). The model forecasts either:
- **`density` target**: the full spectrum E(f) for lead times 6/12/24/48 hours
- **`hs` target**: scalar significant wave height Hs = 4√m₀

Hyperparameter search is managed by Optuna with a SQLite backend (`optuna_study_v2.db`).

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

**Preprocess buoy data** (only needed once; output cached as `buoy_data/processed_data.pkl`):
```bash
python scripts/data_processing.py
```

**Run hyperparameter optimization:**
```bash
python scripts/optimize.py
```

**Plot results** (reads `results/` directory):
```bash
python scripts/plot_results.py
```

## Architecture

### Data pipeline (`utils/`, `nn/prepare_x.py`, `nn/prepare_y.py`)

Raw buoy `.txt` files → `utils/data_processing.py` reads/reindexes/interpolates them → saved to `buoy_data/processed_data.pkl`. The four DataFrames (`density`, `alpha_1`, `alpha_2`, `r_1`) share a full hourly datetime index with shape `(time, num_freqs)`.

Data is split **70 % train / 15 % val / 15 % test** (chronological, no shuffling across splits). `optimize.py` also supports a `deltat` downsampling parameter: `deltat=2` uses every other hourly sample and halves the step count for a given lead time in hours.

**Normalization** (inside `nn/optimization.py::objective()`):
- `density`: scale-only (`E / μ(f)` per-frequency mean). Non-negativity must be preserved because `compute_hs` calls `sqrt(trapz(E))`.
- `alpha_1`, `alpha_2`, `r_1`: z-score.

`freq_means` — the per-frequency training mean μ(f) — is the single tensor that bridges normalized and physical space. It flows through training, evaluation, inference, and the start token.

`prepare_X` stacks the 4 channels into `(samples, seq_len, num_freqs, 4)` tensors. `prepare_y` produces `(samples, lead_time, 1)` for Hs or `(samples, lead_time, num_freqs)` for density.

### Model (`nn/transformer.py`)

`WaveHeightBaselineNN` is a standard encoder-decoder Transformer (`nn.Transformer`). The encoder receives the full multi-channel spectral history; the decoder receives either a scalar Hs start token or the last observed normalised spectrum. Autoregressive inference is in `infer()`.

Embed dim is always `head_dim × nhead` (enforced in `optimization.py`) to guarantee divisibility.

`nn/lstm.py` contains an older encoder-only LSTM baseline (same class name, different interface — no `infer()`, single-step output). It is not wired into the current training/optimization pipeline.

### Training (`nn/training_loop.py`, `nn/optimization.py`)

Training uses teacher forcing with **scheduled sampling**: `tf_ratio` decays linearly from 1.0 → 0.0 over `4 × patience` epochs, closing the teacher-forced/autoregressive gap. For `density` targets, the RMSE loss is computed in **physical space** (denormalized via `freq_means`) so the gradient is not dominated by normalization artifacts.

Early stopping patience=10; LR scheduler `ReduceLROnPlateau` with patience=3 fires before early stopping. Gradients are clipped to `max_norm=1.0` per step.

### Evaluation metrics (`nn/evaluate.py`)

The persistence baseline is the last observed value (start token) broadcast over all lead-time steps. Skill Score `SS = 1 − RMSE_model / RMSE_persistence` is the primary optimization target.

For `density` target with `freq_means`, the evaluator additionally computes bulk parameters: `Hs_RMSE`, `Tm02_RMSE`, `Shape_RMSE` (spectral shape error decoupled from energy magnitude), and `SI_per_bin` (per-frequency Scatter Index). All integrations use `np.trapezoid` on the physical (denormalized) spectra.

### Optuna study (`scripts/optimize.py`)

Study name pattern: `{target}_{deltat_N}_{lead_Nh}_{version}` (e.g. `density_deltat_1_lead_24h_v2`). Results are saved under `results/{target}/deltat_{N}/lead_{N}h/`. Bump `STUDY_VERSION` when the objective function or hyperparameter space changes incompatibly; the old DB file is kept at its old version and a fresh one is created.

Sampler: `TPESampler(n_startup_trials=20, multivariate=True, seed=42)`. Pruner: `MedianPruner(n_warmup_steps=20)`. After each trial, `save_progress` writes a checkpoint to the results folder.

`OBJECTIVE_METRIC` is set per run at the top of `optimize.py` (currently `Shape_RMSE`). Available values: `weighted_mean_SS` (default), `overall_SS`, `RMSE`, `Hs_RMSE`, `Tm02_RMSE`, `Shape_RMSE`, `SI_mean`.

## Key Invariants

- **Always denormalize before physical integrations.** `E_phys = E_norm * freq_means`. Passing `E_norm` directly to `compute_hs` / `compute_bulk_params` produces wrong results — this is explicitly tested in `tests/test_spectral.py`.
- **`freq_means` is fit on training split only**, then applied to val/test splits.
- **Do not call `set_seed()` inside `objective()`** — it collapses per-trial variance that Optuna needs for TPE.
- **Skill Score is not equivalent to RMSE across Optuna trials** because `seq_len` is a hyperparameter, so the persistence baseline varies between trials.
