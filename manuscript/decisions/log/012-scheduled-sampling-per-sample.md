---
status: kept
date: 2026-07-23
commits: [ba8c003]
category: training
---

# 012 — Scheduled-sampling teacher/model choice drawn per-sample, not per-batch

## Context

Scheduled sampling (`nn/training_loop.py::train_one_epoch`) decides, at each autoregressive
decode step, whether the decoder sees the ground-truth previous token or the model's own
previous prediction. The original implementation drew this choice once per batch, so an
entire batch was uniformly "easy" (teacher-forced) or "hard" (autoregressive,
error-compounding) purely by chance — injecting epoch-to-epoch training-signal variance
unrelated to the actual `tf_ratio` schedule.

## Change

The teacher/model choice is now drawn independently per sample (`torch.rand(batch, 1, 1) <
tf_ratio`), so every batch sees a mix of teacher-forced and self-generated context close to
the target `tf_ratio` on average, rather than whole batches falling on one side or the
other by chance.

## Evidence

Reasoning-only: the commit message states the intent ("Should reduce noise during
learning") but no isolated before/after metrics comparison is cited or exists in
`results/` — this predates the `shape_v9`/`v10` studies that would have been the first
candidates for such a comparison, and the change was never isolated from other concurrent
training-loop work.

## Decision

Kept. The statistical argument (per-sample Bernoulli draws concentrate around the target
mix ratio by the law of large numbers within a batch, while a single per-batch draw does
not) is straightforward and uncontested; no regression was observed in subsequent studies.

## Related

Code: `nn/training_loop.py::train_one_epoch`
Manuscript section this might feed: Methods — training procedure (scheduled sampling)
