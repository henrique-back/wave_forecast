# wave_forecast

This project applies transformer-based deep learning to wave spectra forecasting. A buoy produces
4-channel directional wave spectra (spectral density, alpha1, alpha2, r1) over a log-spaced
frequency grid, and the model forecasts significant wave height (`hs`), the full spectrum
(`density`), or the unit-area normalized spectrum shape (`shape`) at lead times of 6/12/24/48
hours.

For the full technical write-up — data pipeline, configurable input channels, normalization,
model architecture, training loop, loss functions, and evaluation metrics — see **[CLAUDE.md](CLAUDE.md)**.
It's written for an AI coding agent but is the canonical, most detailed architecture/pipeline
reference in this repo, kept up to date as the code changes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `torch` to a specific version; if the default `pip install` doesn't give
you the right build for your machine (CPU-only vs. a specific CUDA build), see the comment at the
top of that file.

## Data

Buoy 42056's raw NDBC files (`density.txt`, `alpha1.txt`, `alpha2.txt`, `r1.txt`, `wind.txt`) and
the already-processed `buoy_data/42056/processed_data.pkl` are committed to this repo, so no
external download is required to reproduce results. To (re)process raw data after a change to
`utils/data_processing.py`:

```bash
python scripts/data_processing.py
```

## Reproduce a simple experiment

This runs a short Optuna hyperparameter search for one target/lead time, retrains the best
configuration, and inspects its predictions on the held-out test set.

```bash
source .venv/bin/activate

# 1. Hyperparameter search (edit the config constants at the top of the file
#    first if you want a different target/channel_set/lead_times — the
#    defaults in the repo are a real, currently-active study).
python scripts/optimize.py

# 2. Retrain a final model from the search's best hyperparameters and
#    evaluate it on the held-out test set (edit EXPERIMENT_NAME/target at
#    the top of the file to match what optimize.py just produced).
python scripts/train.py

# 3. Inspect predictions on individual test-set samples (plots + printed metrics).
python scripts/infer.py --experiment <EXPERIMENT_NAME> --target <target> --lead 6
```

Each experiment's results — hyperparameters, validation/test metrics, and (for `density`/`shape`
targets) plots — are written under `results/{EXPERIMENT_NAME}/{target}/lead_{N}h/`. Run
`python scripts/summarize_results.py` to regenerate `results/RESEARCH_LOG.md`, a cross-experiment
comparison table.

## Tests

```bash
pytest tests/
```

Covers the normalize/denormalize round-trip invariants (`tests/test_spectral.py`), the
configurable input-channel axes (`tests/test_channels.py`), the composite directional loss
(`tests/test_loss.py`), and target-specific inference behavior (`tests/test_transformer.py`).

## Repository layout

| Path | Contents |
|---|---|
| `utils/` | Data processing, spectral math (Hs/Tm02/shape), losses, misc helpers |
| `nn/` | Dataset prep, model architecture, training loop, evaluation, Optuna objective |
| `scripts/` | Entry points — data processing, hyperparameter search, final training, inference, results summarization/plotting |
| `tests/` | pytest suite |
| `results/` | Per-experiment hyperparameters, metrics, and plots (committed) |
| `buoy_data/` | Raw and processed buoy data (committed) |

## License

MIT — see [LICENSE](LICENSE).
