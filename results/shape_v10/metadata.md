# Experiment: shape_v10

- **Date**: 2026-07-24
- **Description**: Transformer with convolutional frontend and frequency-structured embedding.Implements attention poolingUses weighted mean Skill Score as objective.Trains to predict spectral shape at 6h, 12h, 24h lead times.Fixes RMSE weighting to use utils.trapz_weights instead of flat mean over log-spaced frequency grid.Adds padding_mode to convolutional frontend and enforce positivity with clamp (at inference) and softplus (at training).Includes r2 as new channel.Switches optimizer to AdamW, narrows lr search range around shape_v9's best trials, and splits the single dropout hyperparameter into freq_embed_dropout and embed_dropout (the latter now also drives nn.Transformer's own internal dropout, previously unwired).
- **STUDY_VERSION**: v10
- **OBJECTIVE_METRIC**: weighted_mean_SS
- **CHANNEL_SET**: full
- **AUX_SET**: none
- **Architecture**: (fill in manually)
