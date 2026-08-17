# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project applies transformer-based deep learning to wave spectra forecasting. A buoy produces 5-channel directional wave spectra (spectral density, alpha1, alpha2, r1, r2) over a log-spaced frequency grid (47 bins, 0.02–0.485 Hz). The model forecasts one of three targets, for lead times of 6/12/24/48 hours:
- **`hs`**: scalar significant wave height Hs = 4√m₀
- **`density`**: the full physical spectrum E(f)
- **`shape`**: the unit-area normalized spectrum E(f)/m₀ — see "Shape/magnitude model split" below

Hyperparameter search is managed by Optuna with a SQLite backend (`optuna_study_{STUDY_VERSION}.db`).

## Commands

All commands should be run from the repo root with the virtualenv active:

```bash
source .venv/bin/activate
```

**GPU-heavy scripts must go through Slurm, not run directly.** wavetank's GPU is shared across the group (netuno is the alternative host, same RAM, GPU usually free) — `scripts/optimize.py`, `scripts/train.py`, and `scripts/ablate_loss.py` all call `utils.require_slurm()` at startup and exit immediately if `SLURM_JOB_ID` isn't set (bypass deliberately, e.g. for a short CPU-only smoke test, with `WAVE_FORECAST_ALLOW_NO_SLURM=1`). Copy `slurm/run.slurm.template` (see `slurm/ablate_*.slurm` for worked examples), fill in `--job-name` and the `python` line, adjust `--time`/`--mem`/`--cpus-per-task` for the run, then:
```bash
mkdir -p logs
sbatch slurm/your_job.slurm     # prints "Submitted batch job <N>"
squeue -u $USER                 # monitor
scancel <N>                     # cancel
```
Chain dependent runs (e.g. a retrain that must wait on a search) with `sbatch --dependency=afterok:<N> ...` — see `slurm/submit_ablation.sh` for a worked multi-job example. Everything below that invokes one of those three scripts assumes it's wrapped in a `.slurm` file this way, even where the command line just shows the bare `python ...` invocation for brevity.

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

**Run hyperparameter optimization** (edit the config constants at the top of the file first — `EXPERIMENT_NAME`, `target`, `CHANNEL_SET`, `AUX_SET`, `OBJECTIVE_METRIC`, `lead_times_hours`; submit via Slurm, see above):
```bash
sbatch slurm/your_optimize_job.slurm    # runs `python scripts/optimize.py`
```

**Retrain a final model** from an Optuna study's best hyperparameters and evaluate on the held-out test set (submit via Slurm, see above):
```bash
sbatch slurm/your_train_job.slurm       # runs `python scripts/train.py`
```

**Run the KL/Wasserstein/peak composite-loss ablation** (`scripts/ablate_loss.py` — fixed architecture pinned from a prior Optuna study, searches only the loss-term weight(s); see its module docstring for the phase sequence and dependency graph). Submit each phase via Slurm; `slurm/submit_ablation.sh` submits and dependency-chains all five in one shot:
```bash
bash slurm/submit_ablation.sh
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

Raw buoy `.txt` files → `utils/data_processing.py` reads/reindexes/interpolates them → saved to `buoy_data/{BUOY_ID}/processed_data.pkl`. Returns a 6-tuple `(density, alpha_1, alpha_2, r_1, r_2, wind)`. The five spectral DataFrames (`density`, `alpha_1`, `alpha_2`, `r_1`, `r_2`) share a full hourly datetime index with shape `(time, num_freqs)` — every buoy folder must provide all five `.txt` files (`density.txt`, `alpha1.txt`, `alpha2.txt`, `r1.txt`, `r2.txt`), `data_processing()` raises `FileNotFoundError` otherwise. `wind` has columns `['wind_u', 'wind_v']` (same hourly index, no freq axis) — derived from `buoy_data/wind.txt` (NDBC stdmet format) by `process_wind()`: NDBC sentinels (`WDIR==999`, `WSPD==99.0`) are masked to NaN, then converted to u/v components (`u = -WSPD·sin(WDIR)`, `v = -WSPD·cos(WDIR)`) **before** time-interpolation, since WDIR is circular and interpolating the raw angle across a gap would pass through the wrong side of the compass.

Data is split **70 % train / 15 % val / 15 % test** (chronological, no shuffling across splits), computed identically inside `nn/optimization.py::_prepare_dataloaders` for both the Optuna search and `scripts/train.py`'s final retrain so their metrics stay comparable.

**Input channels are configurable along two independent axes** (`nn/channels.py`), both threaded through `nn/optimization.py::_prepare_dataloaders`/`objective()` and set in `scripts/optimize.py`/`scripts/train.py`:
- `channel_set` — which frequency-resolved channels are stacked by `prepare_X` into `(samples, seq_len, num_freqs, num_channels)`: `'density'` (density only) or `'full'` (density + alpha_1 + alpha_2 + r_1 + r_2, a 7-channel input — r_1/r_2 together bound the directional spreading function, and r_2/r_1² deviating from 1 flags bimodal/mixed sea states that r_1 alone can't distinguish). `density` is always first — `utils/get_start_token.py` reads channel 0 as the decoder start token.
- `aux_set` — which scalar-per-timestep side channels are fused into the encoder separately (not frequency-resolved, so they bypass `prepare_X`/`FreqDimEmbedding` entirely): `'none'` or `'wind'` (`wind_u`, `wind_v`, windowed by `nn/prepare_aux.py` into `(samples, seq_len, num_aux_channels)`). Wind is encoder-only context — it never reaches the decoder, since it's never a forecast target.

**Normalization** (inside `nn/optimization.py::_prepare_dataloaders`, modes registered in `nn/channels.py`):
- `density`: scale-only (`E / μ(f)` per-frequency mean). Non-negativity must be preserved because `compute_hs` calls `sqrt(trapz(E))`. Always normalized regardless of `channel_set`, since it also produces the `density`-target `y`.
- `alpha_1`, `alpha_2`, `r_1`, `r_2`, `wind_u`, `wind_v`: z-score.

`freq_means` — the per-frequency training mean μ(f) — is the single tensor that bridges normalized and physical space. It flows through training, evaluation, inference, and the start token. For the `hs`/`shape` targets, `prepare_y` builds `y` from the **physical** (pre-normalization) density so the target and persistence baseline are meaningful without needing `freq_means` at loss time; for `density` targets, `y` stays in normalized space and `freq_means` is applied externally at loss/metric time.

### Shape/magnitude model split

Rather than always training one `density`-target model, an alternative is to train **two separate models** at the same lead time — one `hs`-target model (predicts scalar magnitude) and one `shape`-target model (predicts the unit-area spectrum E(f)/m₀, via `compute_shape` in `utils/compute_hs.py`) — and recombine their outputs at inference time:

```
E_pred(f, t) = shape_pred(f, t) * m0_pred(t),   m0_pred = (Hs_pred / 4)^2
```

This decouples the (typically easier) magnitude-forecasting problem from the (typically harder) shape-forecasting problem, motivated by a single `density`-target model tending to underestimate spectral peaks and over-smooth the high-frequency tail. `scripts/infer.py --target combined` loads a matching `hs` checkpoint and `shape` checkpoint (same experiment/lead) and does this recombination for inspection; `scripts/compare_versions.py --experiment name:combined` does the same across the full test set for metric comparison against monolithic `density` models. Example experiments using this split: `hs_shape_v5`, `hs_shape_v6`.

### Model (`nn/transformer.py`)

`WaveHeightBaselineNN` is an encoder-decoder Transformer (`nn.Transformer`, `norm_first=True`) with two custom front-ends:
- **Frequency-structured embedding** (`nn/freq_embedding.py::FreqDimEmbedding`) — instead of flattening `(num_freqs, num_channels)` through one large `Linear` (`nn/embedding.py::Embedding`, only still used for the decoder's scalar `hs` input, which has no frequency axis to structure), each bin's channel measurement is processed and pooled through four stages: (1) a shared linear maps the per-bin channel values into a per-bin representation; (2) a frequency-identity signal is added — a fixed sinusoidal encoding of the bin's actual **log-frequency** value (`_log_freq_sinusoidal_encoding`, wavelengths scaled to the grid's actual span, not the arbitrary base-10000 convention used for long integer time sequences — the grid is non-uniformly spaced, dense near 0.02 Hz and coarse near 0.485 Hz, so embedding distance should track real log-frequency distance, not array position) plus a learned per-bin residual (zero-initialised); (3) a dilated-conv frontend reusing `TemporalConvFrontend` along the frequency axis lets each bin absorb local context from its neighbours, since a wave spectrum is a smooth curve; (4) a single-query attention pool (`_FreqAttentionPool`) collapses all bins into one `embed_dim` token, weighting bins by their actual content (e.g. tracking wherever the spectral peak currently sits) rather than a fixed per-position weight. Both the encoder (`num_channels` = however many `channel_set` selects) and the decoder (`num_channels=1` — the decoder only ever sees a single-channel spectrum) go through `FreqDimEmbedding` for `density`/`shape` targets, so the two sides get the same structural treatment.
- **Temporal conv front-end** (`nn/temporal_conv.py::TemporalConvFrontend`, encoder only) — three dilated `Conv1d` layers (dilation 1, 2, 4) with pre-norm residuals extract local temporal patterns at 3h/5h/9h scales before the global self-attention layers. Not applied to the decoder, which is short (≤ lead_time steps) and already uses a causal mask. The same class is reused *inside* `FreqDimEmbedding` for frequency-axis (rather than time-axis) smoothing — see above.

When `num_aux_channels > 0`, the auxiliary side-input (e.g. wind) is embedded separately via `self.aux_embedding` and added additively into the per-timestep encoder token before positional encoding, since aux channels aren't frequency-resolved. The decoder receives either a scalar Hs start token, a unit-area shape start token, or the last observed normalized spectrum, depending on `target`. Autoregressive inference is in `infer()`, which encodes `src` once and reuses the cached `memory` across all decode steps.

Embed dim is always `head_dim × nhead` (enforced in `optimization.py`) to guarantee divisibility.

**Note on the Meta 4 per-frequency architecture proposal (2026-07-24 meeting doc):** the doc proposes building a per-bin vector `x_k(t) = [E(f_k,t), sinα1(f_k,t), cosα1(f_k,t), sinα2(f_k,t), cosα2(f_k,t), r(f_k,t), f_k]`, passing it through a shared per-frequency MLP `φ` to get `z_k(t) = φ(x_k(t))`, then aggregating via some `Ψ` (linear / frequency-attention / frequency-convolution) into one temporal token. `FreqDimEmbedding` already implements this — and combines two of the three suggested `Ψ` mechanisms rather than picking one: the per-bin `freq_proj` linear + GELU is `φ`; the dilated-conv frontend is a frequency-convolution `Ψ` component; `_FreqAttentionPool` is a frequency-attention `Ψ` component, applied after the conv step. The one literal difference: `f_k` isn't concatenated into `x_k` — it's injected afterward as an additive log-frequency sinusoidal encoding + learned residual, which is a strictly richer frequency-identity signal than a raw scalar, so this is an intentional improvement rather than a gap. The one *real* gap is that this structure currently only benefits the **input** side (`channel_set='full'` on the encoder) — the decoder never predicts `alpha_1`/`alpha_2`/`r_1`/`r_2` (`output_dim` is always `1` or `num_freqs`, see `encode`/`decode` above), so there's no output-side "relate the parameters" benefit yet. `FreqDimEmbedding` is already channel-count-agnostic, so extending it to a future multi-channel directional-prediction target (see `DirectionalLoss` in `utils/loss.py`) would need no changes here — just `num_channels=7` on the decoder side and the target-wiring work described in the Meta 3 discussion.

`nn/lstm.py` contains an older encoder-only LSTM baseline (same class name, different interface — no `infer()`, single-step output). It is not wired into the current training/optimization pipeline.

### Training (`nn/training_loop.py`, `nn/optimization.py`)

Training uses teacher forcing with **scheduled sampling**: `tf_ratio` decays linearly from 1.0 → 0.0 over `4 × patience` epochs, closing the teacher-forced/autoregressive gap. For `density` targets, the RMSE loss is computed in **physical space** (denormalized via `freq_means`) so the gradient is not dominated by normalization artifacts.

`nn/optimization.py::_train_model` is shared by `objective()` (Optuna trial) and `scripts/train.py` (fixed-config final retrain) so the two never drift apart — the only behavioral difference is that a `trial` object, when passed, reports per-epoch scores to Optuna and can prune the run.

Early stopping patience=20 (Optuna trials) / configurable in `scripts/train.py`; LR scheduler `ReduceLROnPlateau(patience=5, factor=0.5, cooldown=2)` fires before early stopping. Gradients are clipped to `max_norm=1.0` per step.

### Evaluation metrics (`nn/evaluate.py`)

The persistence baseline is the last observed value (start token) broadcast over all lead-time steps. Skill Score `SS = 1 − RMSE_model / RMSE_persistence` is the primary optimization target family.

`evaluate()` always returns `Hs_SS` (for `target == 'hs'` this equals `overall_SS` exactly; for `density` it's derived from denormalized spectra) — robust to `seq_len` variation, since the persistence baseline's RMSE changes with `seq_len` but Hs itself does not.

For `density` target with `freq_means`, the evaluator additionally computes bulk parameters: `Hs_RMSE`/`Hs_Bias`, `Tm02_RMSE`/`Tm02_Bias`, `Shape_RMSE` (spectral shape error decoupled from energy magnitude, masked below `M0_MASK_THRESHOLD`), and `SI_per_bin`/`SI_mean` (per-frequency Scatter Index). All integrations use `np.trapezoid` on the physical (denormalized) spectra.

### Spectral peak detection / partitioning (`utils/spectral_partitioning.py`, `utils/spectral_peaks.py`)

`utils/spectral_partitioning.py::find_significant_peaks` is the peak detector: the Portilla et al. (2009, section 2b.2) four-criterion significant-peak test, applied to a single 1-D spectrum against its physical frequency grid `freqs` — a peak is rejected as spurious if any of: (1) `fp > f_max` (default 0.4 Hz, high-frequency tail noise), (2) its trough-to-trough partition energy is below `energy_frac` (default 0.05) of the spectrum's total energy, (3) it has fewer than `min_bins` (default 2) spectral bins on either side before the next trough, or (4) it is "sandwiched" between two higher-energy neighboring peaks. This replaced an earlier scale-free `prominence_frac` heuristic (relative to each spectrum's own max, tuned empirically by sweeping the constant against one buoy's test set) with published, physically-motivated criteria. The same module also has `classify_partition`/`classify_partitions`, which label a detected peak `'wind_sea'` vs `'swell'` via `γ* = S_obs(fp) / S_PM(fp)` (observed energy at the peak vs. the Pierson-Moskowitz reference spectrum evaluated analytically at that frequency, threshold 1.0 per Violante-Carvalho 2009). Wired into `nn/evaluate.py` via `utils/spectral_peaks.py::peak_modality_metrics` (see below) for a wind-sea/swell-conditioned metric breakdown.

`utils/spectral_peaks.py` is the batch-metrics layer built on top: `find_spectral_peaks(freqs, spectrum, f_max, energy_frac, min_bins)` is a thin wrapper around `find_significant_peaks` (with a `spectrum.max() <= 0` short-circuit for degenerate/all-zero spectra), and `peak_modality_metrics(freqs, pred_final, true_final, ...)` is the batched, JSON-safe metric used by `nn/evaluate.py` (`target == 'shape'`, `compute_peak_metrics=True`) to surface the multimodal (double/triple-peaked) sea-state failure mode that a whole-spectrum frequency-weighted RMSE dilutes away — see its docstring for `Peak_Count_*_Mean`/`Peak_Height_RelError`/`Peak_Separation_Recall` and the `multimodal_mask` it returns alongside. Every true partition (from `find_peak_windows`) is additionally classified `'wind_sea'`/`'swell'` via `classify_partition`, and `Peak_Height_RelError`/`Peak_Separation_Recall` are pooled a second time per label (`_windsea`/`_swell` suffixes) alongside a new `Tm02_RMSE_windsea`/`Tm02_RMSE_swell` (Tm02 = √(m₀/m₂), integrated within each true partition's own trough-to-trough window rather than the whole spectrum) — motivated by the same dilution problem `peak_modality_metrics` already exists to fix, one level down: a single pooled number blends wind-sea partitions (broad, energetic, fast-evolving — a magnitude/energy-tracking problem) with swell partitions (narrow, slow, persistent — closer to a position/shift problem, e.g. what `SpectralWassersteinLoss` specifically targets), hiding which failure mode a given loss change actually fixed. Partition geometry and labels always come from the TRUE spectrum only, applied to both sides — same convention as `SoftPeakHeightLoss`'s true-window-only `tau_k`/`H_true`. `nn/evaluate.py` also computes a whole-spectrum `Tm02_RMSE`/`Tm02_Bias` for the `shape` target unconditionally (no `compute_peak_metrics` gate needed — cheap, no scipy loop) via `compute_bulk_params` on the shape spectrum directly: Tm02 is scale-invariant (m₀/m₂ ratio unaffected by a uniform rescaling of E(f)), so it's physically meaningful without `freq_means`, unlike Hs. Both spectra must be physical/linear (already `exp()`'d out of log-space) and on the same `freqs` grid, compared by aligned bin index. `scripts/plot_cdf_wasserstein.py` also uses `find_spectral_peaks` to bucket test samples into unimodal/multimodal representatives for its CDF/Wasserstein visualization.

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
