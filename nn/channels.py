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
    # alpha_1/alpha_2 (mean/principal wave direction, degrees) are circular,
    # so they're fed as sin/cos pairs rather than the raw angle — otherwise
    # 1deg and 359deg would appear maximally far apart to the model.
    'full':    ['density', 'alpha_1_sin', 'alpha_1_cos', 'alpha_2_sin', 'alpha_2_cos', 'r_1', 'r_2'],
}

NORM_MODES = {
    'density':     'scale',  # non-negativity required for compute_hs / sqrt
    'alpha_1_sin': 'none',   # already in [-1, 1]; z-scoring would distort a valid unit circle
    'alpha_1_cos': 'none',
    'alpha_2_sin': 'none',
    'alpha_2_cos': 'none',
    'r_1':         'zscore',
    'r_2':         'zscore',
}

# Computed once per sample from that sample's own history (nn/prepare_dmd.py),
# then broadcast across seq_len to fit prepare_aux's (samples, seq_len,
# channels) contract — see manuscript/decisions/log/019. Column count must
# stay in sync with nn.prepare_dmd.DEFAULT_N_MODES (4 modes x 3 features/mode
# = 12), same tight-coupling convention as utils/loss.py::_FULL_CHANNELS.
_DMD_COLUMNS = [
    f'dmd_mode{k}_{feat}' for k in range(4) for feat in ('growth', 'freq', 'amp')
]

AUX_CHANNEL_SETS = {
    'none': [],
    'wind': ['wind_u', 'wind_v'],
    'dmd':  _DMD_COLUMNS,
}

AUX_NORM_MODES = {
    'wind_u': 'zscore',
    'wind_v': 'zscore',
    **{name: 'zscore' for name in _DMD_COLUMNS},
}
