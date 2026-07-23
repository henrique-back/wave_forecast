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
    # alpha_1/alpha_2 are circular (mean/principal wave direction, degrees) —
    # fed as sin/cos pairs rather than the raw angle so that e.g. 1deg and
    # 359deg are adjacent to the model instead of maximally far apart.
    'full':    ['density', 'alpha_1_sin', 'alpha_1_cos', 'alpha_2_sin', 'alpha_2_cos', 'r_1', 'r_2'],
}

NORM_MODES = {
    'density':     'scale',  # non-negativity required for compute_hs / sqrt
    'alpha_1_sin': 'none',   # already in [-1, 1]; z-scoring would just distort a valid unit circle
    'alpha_1_cos': 'none',
    'alpha_2_sin': 'none',
    'alpha_2_cos': 'none',
    'r_1':         'zscore',
    'r_2':         'zscore',  # same rationale as r_1 — no physical non-negativity constraint
}

AUX_CHANNEL_SETS = {
    'none': [],
    'wind': ['wind_u', 'wind_v'],
}

AUX_NORM_MODES = {
    'wind_u': 'zscore',
    'wind_v': 'zscore',
}
