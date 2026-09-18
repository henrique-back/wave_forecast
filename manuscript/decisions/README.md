# Decision log

This directory records *why* the model, data pipeline, and evaluation methodology ended up
the way they did — not just what the code currently does. Most of the project's development
was ablation-driven (implement a change → run it → compare against the current best → keep
or revert), so most entries are short records of one such step. Read this before writing any
methodological justification in the manuscript itself (Methods, Discussion, reviewer
response) — see `../CLAUDE.md` section 4 for that usage rule.

## Two tiers

**`log/`** — one short Markdown file per ablation/design decision, using
[`log/TEMPLATE.md`](log/TEMPLATE.md). This is the default tier. Every ablation step that
changed the model, data, training loop, or evaluation and produced a keep/revert call
belongs here, however small. Numbered by rough chronological order of the underlying
decision (not the order files were written), `NNN-slug.md`.

**Top-level `.tex` docs** — a full standalone LaTeX document (own title, sections,
bibliography), for a decision that needs citation-backed prose because it will be argued
in the manuscript itself, not just recorded for internal reference. Promote a `log/` entry
to one of these when you actually start drafting the corresponding Methods/Discussion
paragraph and it needs more than a paragraph of justification. Current entry:

- [`wasserstein_kl_justification.tex`](wasserstein_kl_justification.tex) — composite
  spectral loss (RMSE + KL-divergence + Wasserstein-1 + soft-max peak height).

Most decisions will never need promotion — the `log/` entry is the final record for them.

## Index

| # | Decision | Status | Date |
|---|----------|--------|------|
| [001](log/001-optuna-objective-mean-step-skill-score.md) | Optuna objective: teacher-forced RMSE → mean per-step Skill Score | Kept | 2026-04-08 |
| [002](log/002-freq-structured-embedding-and-temporal-conv.md) | Encoder front-end: flat `Linear` embedding → `FreqDimEmbedding` + `TemporalConvFrontend` | Kept | 2026-07-03 |
| [003](log/003-hs-shape-model-split.md) | Two separate `hs`+`shape` models recombined at inference, vs. one monolithic `density` model | Kept | 2026-07-07 |
| [004](log/004-wind-auxiliary-channel.md) | Add `wind_u`/`wind_v` as an encoder-only auxiliary input | Kept, marginal | 2026-07-06 |
| [005](log/005-median-pruner-warmup-steps.md) | Optuna `MedianPruner`: `n_warmup_steps` 20 → 30 | Kept | 2026-07-13 |
| [006](log/006-freqdimembedding-conv-attention-pool.md) | `FreqDimEmbedding` internals: flatten+`Linear` pool → dilated-conv + attention pool | Kept | 2026-07-14 |
| [007](log/007-linear-baseline-sanity-check.md) | Per-frequency linear regression baseline vs. the transformer | Kept as ongoing baseline | 2026-08-05 |
| [008](log/008-peak-detection-portilla-criteria.md) | Peak detector: scale-free `prominence_frac` heuristic → Portilla et al. (2009) 4-criterion test | Kept | 2026-08-06 |
| [009](log/009-peak-fidelity-objective-metric.md) | `peak_fidelity_SS` as an Optuna objective metric | Exploratory — open | 2026-08-21 |

Composite spectral loss (KL + Wasserstein + soft-max peak height) — see
[`wasserstein_kl_justification.tex`](wasserstein_kl_justification.tex) directly; it was
never a short `log/` entry.

## Adding an entry

1. Copy `log/TEMPLATE.md` to `log/NNN-slug.md`, next number in the index above.
2. Fill it in from primary sources only — `results/RESEARCH_LOG.md`, a specific
   `results/{EXPERIMENT}/.../metrics.json` or `results/comparisons/.../metrics.json`, or the
   commit that made the change. If a clean isolated before/after number doesn't exist
   (common — real ablations often bundle more than one change in the same commit/study),
   say so explicitly in the entry rather than presenting a confounded comparison as if it
   isolated one variable.
3. Add a row to the index table above.
4. If a future finding supersedes an entry, don't delete it — add a `Superseded by: NNN`
   line to its Status and leave the historical reasoning in place.
