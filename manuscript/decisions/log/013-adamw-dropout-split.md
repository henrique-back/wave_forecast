---
status: kept
date: 2026-07-24
commits: [7afb843]
category: training
---

# 013 — Optimizer Adam → AdamW; single `dropout` split into `freq_embed_dropout`/`embed_dropout`

## Context

Training used `torch.optim.Adam`, whose L2 weight decay is coupled with the gradient
update (unlike AdamW's decoupled decay). Separately, a single `dropout` hyperparameter
covered every dropout site in the model, and `nn.Transformer`'s own internal
attention/FFN dropout was never wired up at all — it was silently stuck at PyTorch's
default of 0.1 regardless of what `dropout` was set to.

## Change

Switched the optimizer to `optim.AdamW` (`nn/optimization.py::_train_model`) and split
`dropout` into two Optuna-tunable hyperparameters: `freq_embed_dropout`
(`FreqDimEmbedding`'s internal `freq_embed_dim=8`-wide conv) and `embed_dropout`
(`PositionalEncoding`, the top-level time-axis `TemporalConvFrontend`, and — newly wired —
`nn.Transformer`'s own internal dropout). `lr`'s search range was narrowed to bracket
`shape_v9`'s best trials (1e-3–1.5e-2, was 1e-4–1e-2). `weight_decay`'s range was
deliberately left unchanged despite the optimizer switch, since AdamW's decoupled decay
behaves differently for the same numeric value and `shape_v9`'s Adam-tuned values weren't
known to transfer.

## Evidence

`results/shape_v9/shape/lead_{6,12,24}h/best_trial.txt` show a single `dropout` value
(0.13–0.27) and `lr` in 0.0037–0.0092 under the old regime.
`results/shape_v10/shape/lead_{6,12,24}h/best_trial.txt` show the new split
(`freq_embed_dropout`, `embed_dropout`) and confirm `lr` landing within the narrowed
1e-3–1.5e-2 range at every lead time. No isolated same-architecture Adam-vs-AdamW
comparison exists — this was not bundled as a controlled ablation, and the version bump
(v9→v10, no DB rename per the code comment) reflects that old v9 trials were judged
incomparable rather than a clean before/after being preserved.

## Decision

Kept. The previously-unwired `nn.Transformer` internal dropout was a real bug (silently
fixed at 0.1 regardless of the tuned value), and AdamW's decoupled weight decay is the
standard fix for Adam's known coupling issue — this was treated as a correctness/best-
practice update rather than something requiring its own ablation.

## Related

Code: `nn/optimization.py::_train_model`, `nn/optimization.py::objective`
Other decisions: [[006]] (`FreqDimEmbedding` internals, including `freq_embed_dim`)
Manuscript section this might feed: Methods — training procedure / hyperparameter search
