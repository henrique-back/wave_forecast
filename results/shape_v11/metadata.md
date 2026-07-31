# Experiment: shape_v11

- **Date**: 2026-07-28
- **Description**: Transformer with convolutional frontend and frequency-structured embedding.Implements attention poolingUses final-step Skill Score as objective (last forecast step only, not an average across autoregressive steps).Trains to predict spectral shape at 6h, 12h, 24h lead times.Fixes RMSE weighting to use utils.trapz_weights instead of flat mean over log-spaced frequency grid.Adds padding_mode to convolutional frontend.Includes r2 as new channel.Switches optimizer to AdamW, narrows lr search range around shape_v9's best trials, and splits the single dropout hyperparameter into freq_embed_dropout and embed_dropout (the latter now also drives nn.Transformer's own internal dropout, previously unwired).metric computed only on lead time step of interest, not averaged across autoregressive steps.v11: predicts log-spectral-energy (log E(f)/m0) directly via a plain Linear head instead of a Softplus-activated linear value; loss switched to frequency-weighted plain MSE in log-space; non-negativity of the physical shape now comes from exp() at inference/metric time instead of an architectural Softplus constraint.
- **STUDY_VERSION**: v11
- **OBJECTIVE_METRIC**: final_step_SS
- **CHANNEL_SET**: full
- **AUX_SET**: none
- **Architecture**: (fill in manually)
