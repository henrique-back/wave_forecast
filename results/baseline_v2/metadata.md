# Experiment: baseline_v2

- **Date**: 2026-07-01
- **Description**: Baseline transformer with flat encoder embedding (Embedding class). No frequency-structured embedding, no temporal conv frontend. Objective optimized on Shape_RMSE. All 4 lead times (6h, 12h, 24h, 48h). Note: 48h best_trial.txt not written (optimization was cut short).
- **STUDY_VERSION**: v2
- **OBJECTIVE_METRIC**: Shape_RMSE
- **Architecture**: WaveHeightBaselineNN — flat Linear(num_freqs×num_channels, embed_dim) encoder embedding, standard Transformer encoder-decoder, no temporal conv
