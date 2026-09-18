---
status: kept
date: 2026-08-05
commits: [fc9e9f6]
category: data
---

# 017 — `BUOY_ID` silently drifted to the wrong buoy (`'42056'`) in `scripts/compare_versions.py`; fixed there

## Context

Every checkpoint-producing script (`scripts/optimize.py`, `scripts/train.py`,
`scripts/infer.py`, `scripts/train_linear_baseline.py`) trains against buoy `'32012'`
(NDBC 32012, the project's single-buoy study — see root `CLAUDE.md`). A prior commit had
silently changed `scripts/compare_versions.py`'s `BUOY_ID` to `'42056'` with no comment or
rationale, meaning every comparison run through that script was scored against the wrong
buoy's data.

## Change

`scripts/compare_versions.py`'s `BUOY_ID` was corrected back to `'32012'`, with a comment
added explaining the requirement (must match the buoy every checkpoint-producing script
actually trains on) and naming the prior drift explicitly so it isn't silently
reintroduced.

## Evidence

`git log -S"a prior commit had silently drifted" -- scripts/compare_versions.py` traces the
fix to commit `fc9e9f6` ("linear baseline"). No `results/comparisons/` output run against
the wrong buoy was identified as needing retraction — this decision only documents the bug
and its fix in `compare_versions.py`, not a broader audit of every comparison ever run.

## Decision

Kept (fixed) in `scripts/compare_versions.py`. **However, this exact bug has since
recurred**: as of this writing, `scripts/data_processing.py:22` hardcodes
`BUOY_ID = "42056"`, again inconsistent with every other script and with root `CLAUDE.md`.
This was flagged to the author during the comment-cleanup pass that produced this entry
(2026-09-18) but deliberately left unfixed pending the author's own review — see the
conversation that produced this entry for context. Anyone relying on
`scripts/data_processing.py` to regenerate `processed_data.pkl` should check `BUOY_ID`
before running it.

## Related

Code: `scripts/compare_versions.py`, `scripts/data_processing.py` (currently regressed —
see Decision above)
Manuscript section this might feed: none (data-hygiene bug, not a methodological choice) —
but any manuscript claim tracing a number to a `compare_versions.py` run predating `fc9e9f6`
should be double-checked for which buoy it actually used.
