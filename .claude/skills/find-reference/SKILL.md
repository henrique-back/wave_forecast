---
name: find-reference
description: "Find and file one or two literature references for a topic, claim, or framing the author wants to write about in the manuscript (e.g. '/find-reference AI wave forecasting has focused on scalar params not the spectrum'). Runs this project's literature-triage convention end to end: reduce the topic to a search query, check the existing corpus first, search further only if there's a gap, verify metadata independently, then file the result into manuscript/literature/notes.md, notes/<bibkey>.md, refs.bib, and personal_notes.md. Use whenever the author has a topic in mind and wants a citation found and filed, not just discussed."
---

# find-reference

Turns "I want to write about X" into one or two properly-filed citations, using the
same triage convention already established in `manuscript/literature/` (see that folder's
`notes.md` header and `manuscript/CLAUDE.md` §3). Read both before the first run in a session
if you haven't already — this skill assumes their rules and doesn't repeat them all here.

## Inputs

`args` is the author's topic/claim in their own words — a full sentence is fine, e.g. "how
improvements in ocean forecasting using statistical/AI models has focused on scalar parameters
not the spectrum." Don't ask the author to pre-format it as a search string; that's step 1.

## Procedure

### 1. Reduce to a search string

Distill the topic into 2-4 short candidate search queries using domain vocabulary, not the
author's own phrasing verbatim — e.g. the example above becomes something like:
- `significant wave height forecasting neural network bulk parameters`
- `spectral shape prediction deep learning wave spectrum`
- `Hs forecasting vs spectral density forecasting machine learning`

Keep them narrow enough to be useful for both the local grep and a web search.

### 2. Check the existing corpus first

Before searching the web, grep what's already vetted — a lot of ground is already covered
(the 2026-09-18 triage pass alone covers ~30 papers):
```
grep -il "<keyword>" manuscript/literature/notes/*.md
grep -i "<keyword>" manuscript/literature/notes.md manuscript/literature/personal_notes.md
```
Also skim the "Web-only comparators", "The gap this corpus does not fill", and "Pending
verification" sections of `notes.md` — the topic may already be a documented gap or a
half-verified lead rather than something to search fresh.

If one or two existing entries already fit well (e.g. for the scalar-vs-spectrum example:
`naithani2005ann`, `sakhare2009svr`, `mandal2005bpnn`, `namekar2006ann` are all
params→shape/scalar-only precedents already in the corpus) — **stop here**, skip to step 5,
and report those directly. Don't do a redundant web search when the corpus already has it.

### 3. Search further only if there's a genuine gap

If nothing in the corpus fits, WebSearch using the queries from step 1. Prefer papers that:
- are peer-reviewed (journal/conference), not blog posts or preprints without a DOI where a
  published version exists;
- are close analogs even if not a perfect match — say so explicitly, the way `notes.md`
  hedges its "closest analog" entries (e.g. "transformer, multi-horizon — but joint scalar
  variables, not spectrum").

Pick at most one or two candidates. More than that is triage work for a dedicated pass, not
this quick-lookup flow — tell the author if the topic looks like it needs a fuller pass instead.

### 4. Verify metadata independently — never invent it

Per `manuscript/CLAUDE.md` §3 ("never cite from training-data recall"): confirm
author list, year, venue, volume/pages, and DOI from the search result or the paper's own
landing page — not from model memory. If a DOI or page range can't be found, leave it out of
`refs.bib` rather than guessing (follow the `meng2023windswell`-style entry — no `doi` field —
for unresolved cases). If metadata stays incomplete, file it under "Pending verification — do
NOT cite yet" in `notes.md` instead of adding a `refs.bib` entry.

### 5. File the result

For each reference kept (whether from step 2 or step 3):

1. **`manuscript/literature/notes/<bibkey>.md`** — create if new, following the exact template
   already in that folder (frontmatter: `key`, `title`, `authors`, `year`, `venue`, `relevance`,
   `source_file` — use "web search, no local PDF" for `source_file` if there's no PDF in hand;
   body: `## Summary`, `## Relevance to this manuscript`, `## Suggested use`).
2. **`manuscript/literature/notes.md`** — add a row to the Index table (or the "Web-only
   comparators" table if no local PDF), and, if it closes or narrows a gap described in
   "The gap this corpus does not fill", update that section too.
3. **`manuscript/literature/refs.bib`** — add a BibTeX entry matching the existing style
   (`@article{...}` with `author`/`title`/`journal`/`volume`/`pages`/`year`/`doi`, alphabetical
   fields as in existing entries) only if metadata is independently confirmed (step 4).
4. **`manuscript/literature/personal_notes.md`** — append one entry under `## Entries` using
   the file's own template, tagged `Reference: <bibkey>`, capturing the specific angle the
   author wants to use this for (their original topic/claim from `args`) — this is what makes
   the reference retrievable by framing later, not just by keyword.

Do not touch any `.tex` file in this step — filing a reference is not the same as citing it in
the manuscript prose; that's a separate, deliberate edit per `manuscript/CLAUDE.md` §8.

### 6. Report back

Give the author, in a few lines:
- the bibkey(s) chosen and a one-sentence reason each fits their topic;
- whether it came from the existing corpus or a fresh search;
- anything hedged or incomplete (no DOI, closest-analog rather than exact match, pending
  verification) — flag it the same way `notes.md` already does, don't smooth it over;
- which four files were touched.

Keep this to a short summary — the filed notes carry the detail, the chat reply doesn't need
to repeat it.
