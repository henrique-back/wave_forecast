# Experiment: shape_v8

- **Date**: 2026-07-16
- **Description**: Transformer with convolutional frontend and frequency-structured embedding.Implements attention poolingUses weighted mean Skill Score as objective.Trains to predict spectral shape at 6h, 12h, 24h lead times.Fixes RMSE weighting to use utils.trapz_weights instead of flat mean over log-spaced frequency grid.
- **STUDY_VERSION**: v8
- **OBJECTIVE_METRIC**: weighted_mean_SS
- **CHANNEL_SET**: full
- **AUX_SET**: none
- **Architecture**: (fill in manually)
