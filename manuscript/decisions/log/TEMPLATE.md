---
status: kept | reverted | superseded (by NNN) | exploratory
date: YYYY-MM-DD
commits: [<short hash>, ...]
category: architecture | training | objective-metric | evaluation | data
---

# NNN — <short decision title>

## Context

What problem or observation motivated looking at this. Quote the specific symptom if there
is one (a metric plateau, a diagnosed failure mode, a reviewer/collaborator question) —
don't reconstruct a plausible-sounding motivation after the fact.

## Change

What was actually implemented, in a sentence or two. Link the relevant file(s)/function(s)
rather than pasting code.

## Evidence

The before/after comparison that backed the keep/revert call, with its source cited
(`results/RESEARCH_LOG.md` row, a specific `metrics.json` path, or a commit message if the
change was structural/methodological rather than metric-driven). If the change was bundled
with others in the same commit/study and the comparison isn't a clean single-variable
ablation, say so explicitly — a confounded comparison honestly labeled is more useful than
a clean-looking one that overclaims.

## Decision

Kept / reverted / still open, and the one-line reason. If reverted or superseded, say what
replaced it and link that entry.

## Related

Code: `path/to/file.py`
Other decisions: [[NNN]]
Manuscript section this might feed (if any):
