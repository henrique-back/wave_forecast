"""Registries for the configurable input-channel axes.

Two independent axes select the model's encoder input:

- `channel_set` : which frequency-resolved channels (shape (time, num_freqs))
  are stacked into `prepare_X`. `density` must always be first in every
  list — `utils/get_start_token.py` reads channel 0 of `src` as the
  (normalised) spectral density for both the 'hs' and 'density' targets.
- `aux_set`     : which scalar-per-timestep side channels (e.g. wind) are
  fused into the encoder token via `WaveHeightBaselineNN`'s aux embedding.
  These are NOT frequency-resolved, so they never go through `prepare_X` /
  `FreqDimEmbedding` — see `nn/prepare_aux.py`.
"""

CHANNEL_SETS = {
    'density': ['density'],
    'full':    ['density', 'alpha_1', 'alpha_2', 'r_1'],
}

NORM_MODES = {
    'density': 'scale',   # non-negativity required for compute_hs / sqrt
    'alpha_1': 'zscore',
    'alpha_2': 'zscore',
    'r_1':     'zscore',
}

AUX_CHANNEL_SETS = {
    'none': [],
    'wind': ['wind_u', 'wind_v'],
}

AUX_NORM_MODES = {
    'wind_u': 'zscore',
    'wind_v': 'zscore',
}
