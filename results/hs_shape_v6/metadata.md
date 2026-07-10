# Experiment: hs_shape_v6

- **Date**: 2026-07-10
- **Description**: Transformer with convolutional frontend and frequency-structured embedding.Uses weighted mean Skill Score as objective.Trains to predict Hs and spectral shape (density target) at 6h, 12h, 24h, 48h lead times.Increases patience to 20 epochs, and n_warmup_steps to 40, so early stopping is more robust to noise and the model has more time to benefit from LR reductions.
- **STUDY_VERSION**: v6
- **OBJECTIVE_METRIC**: weighted_mean_SS
- **CHANNEL_SET**: full
- **AUX_SET**: none
- **Architecture**: (fill in manually)
