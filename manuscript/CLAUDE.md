# CLAUDE.md — manuscript/

Instructions for working in this folder. Read this before touching any `.tex` file here.

## 0. Manuscript narrative spine (main idea, gap, contributions)

Settled 2026-09-23, synthesized from `literature/personal_notes.md` and the "gap this corpus
does not fill" section of `literature/notes.md` — read this before drafting or revising
`sections/01_introduction.tex` (and before any Discussion framing that restates the paper's
contribution). Treat this as the agreed story to write *to*, not a first draft to reconstruct
each time; revise it here if the story changes rather than silently drafting a different one
in `01_introduction.tex`.

### 0.1 The headline — which story this paper tells

**Settled: the paper leads with the measurement/objective argument, not with the model.**

The headline is *"the way spectral wave forecasts are trained and evaluated systematically
rewards over-smoothing, and here is a loss and an evaluation panel that do not"*. The transformer
is the vehicle that demonstrates it, not the claim itself.

This choice is deliberate and should not be quietly reversed while drafting. Framing the paper as
"a transformer that forecasts wave spectra" invites exactly the critique in `jiang2024comment`
(complex models do not outperform auto-regression for Hs time series) and positions the work as
one more deep-learning application. The measurement argument is the more transferable and the
more defensible contribution, and it is what the loss ablation actually demonstrates.

### 0.2 Main idea

A transformer forecasts the 1D wave spectrum from a single buoy's own recent history (5
channels), rather than only a scalar bulk parameter (Hs or Tp) — either directly (`density`
target) or via a shape/magnitude decomposition (separate `hs` and `shape` models recombined at
inference; see parent `../CLAUDE.md`, "Shape/magnitude model split") that decouples the easier
magnitude-forecasting problem from the harder shape-forecasting one. Both spectral targets are
predicted in **log space**, so non-negativity of the recovered physical spectrum follows by
construction (exponentiation at the point of use) rather than from a saturating output activation
or clipping — a small but real modelling choice, already written up in `02_methods.tex`
§Problem formulation, and worth one clause in the contributions list.

### 0.3 The gap, four layers deep (each sits inside the one before it)

1. *AI/DL vs numerical models, generally.* Data-driven models — transformers especially — now
   **match or approach** operational numerical weather prediction skill for meteorological
   forecasting; the analogous gains have been slower to materialise for ocean forecasting.
   Cite `rasp2024weatherbench2` (WeatherBench 2) and `benbouallegue2024rise` (ECMWF's
   operational-like assessment).
   **WORDING CONSTRAINT — do not write "consistently outperform".** Neither source supports it;
   `benbouallegue2024rise` reports "comparable accuracy" and names drawbacks explicitly. Both are
   also **atmosphere-only**, so the "slower for ocean forecasting" half is still uncited — hedge
   it as the authors' own observation. See `personal_notes.md`, "weather - numerical vs AI
   models", for the full correction and the GraphCast alternative if a stronger verb is ever
   wanted.
2. *Within ocean forecasting, AI/ML work has concentrated overwhelmingly on scalar bulk
   parameters.* The scalar lineage runs from 2004 to 2026, and none of its steps forecast the
   full spectrum:
   - early ANNs forecasting Hs/Tz from a buoy's own record at 1–24 h (`makarynskyy2004`). Its
     "correction" networks correct the ANN's own initial forecasts, **not** a numerical model's;
   - EMD-LSTM (`zhou2021emdlstm`, Hs at 3–72 h);
   - the most recent transformer work (`kim2026metoformer`, joint scalar metocean variables).

   The consistency of the pattern across 22 years is what makes it a real, persistent gap rather
   than an artifact of older literature.

   **Do not write that no learned model forecasts the spectrum** (updated 2026-09-29). Learned
   spectral work exists, but it addresses different problems:
   - estimation from bulk parameters or from platform response (`naithani2005ann`,
     `namekar2006ann`, `sakhare2009svr`, `nielsen2023shipbuoy`, `kwon2026transformer`);
   - spatial prediction from atmospheric forcing (`song2023jpo`, `gao2025jgr`,
     `liu2025cnnxlstm`, `cao2025bohai`);
   - post-processing a numerical model's spectral forecast into a buoy-consistent 1D spectrum out
     to 5 days (`filoche2026postprocessing`).

   The defensible gap is the narrower one `01_introduction.tex` already states: forecasting the
   spectrum forward in time from a station's own recent spectral history, with no numerical-model
   input.

   Cite `zhou2021emdlstm` and `ding2023eofeemd` for the lineage only. `jiang2024comment` (its
   ll. 248–249) names both as decomposing the signal across the train/test boundary, so their
   error figures are not usable as comparators.
3. *Bulk parameters are not a sufficient statistic for what the forecast is used for.* This is the
   **consequence** argument, and it must sit alongside the information argument (the spectrum is
   richer; bulk parameters are derivable from it but not the reverse) — on its own the information
   argument invites "richer for what?". Structural and vessel response is frequency-dependent, so
   two sea states carrying identical total energy — identical Hs — produce materially different
   motions, loads and fatigue when that energy is distributed differently across frequency.
   Cite `orimoloye2019bimodal`, which constructs **energy-conserved** bimodal sea states (total
   energy fixed, swell fraction and swell peak period varied) and finds the wave-height
   distribution changing with them.
   **Careful with direction:** that paper reports *greater* non-linearity in unimodal seas at
   matched energy. The citable claim is "spectral composition changes behaviour at fixed Hs", NOT
   "bimodal seas are more severe". `simao2025bimodal` would carry this through to moored-structure
   extreme response, which is the step we actually want — but **it is unread** (ASME paywall), so
   attribute nothing to it until someone reads it.
   This layer is also the motivation for the multimodal/partition-conditioned evaluation panel:
   bimodality is not a completeness exercise, it is precisely the regime where bulk parameters are
   least informative, and therefore where whole-spectrum aggregate scoring is most misleading.
4. *Forecasting the spectrum breaks the metrics used for the scalar problem — the headline layer.*
   A frequency-weighted aggregate error over the whole spectrum dilutes an error confined to a
   narrow peak (1-2 of 47 bins) almost to invisibility, so a forecast that merges a distinct swell
   and wind-sea pair into one smoothed hump can score as well as a faithful one. Compounding this,
   models trained under different loss terms cannot be compared on either one's own objective — a
   model optimised for RMSE will naturally score better on RMSE than one optimised for Wasserstein,
   and neither comparison establishes which generalises better across the whole spectrum. Hence
   both a peak-resolved, partition-conditioned evaluation panel and a loss-agnostic selection
   criterion.
   Sourcing for this layer: `hernandez2025intercomparison` for the present-day scale of operational
   wave verification (WMO Lead Centre, 18 systems), `ecmwf2026lcwfvproject` for its verification
   against buoy data, and `ecmwf2019lcwfvparameters` for what it verifies (six integrated
   parameters derived from the 2-D spectrum; the spectrum itself is not exchanged). Hernandez does
   not name the variables, so the Parameters page carries that claim. `bidlot2002intercomparison`
   is no longer cited here. Then `hanson2009pacific` for the
   wave-modelling community's own published critique of it — "one must look into the spectral
   details to identify sources of model deficiencies", bulk parameters "can mask higher-order
   deficiencies". `benbouallegue2024rise` supplies independent cross-domain evidence that
   **over-smoothing is a characteristic failure mode of data-driven forecasters**, which reframes
   this from a wave-specific quirk into an instance of a documented pathology.

   Two wave-domain sources back this up:
   - `filoche2026postprocessing` (§4.2.1) is a learned spectral forecast whose authors *name the
     mechanism*. Ensemble averaging "may also smooth narrow or highly variable wind-sea peaks",
     and together with "the relative weighting induced by the logarithmic MAE" it may explain
     their conservative wind-sea estimates.
   - `minuzzi2023lstm` reports smoothing toward the mean for scalar Hs. It is already cited
     beside `benbouallegue2024rise` in `01_introduction.tex`.

   Keep the Filoche attribution as hedged as its authors keep it ("may contribute"; "the
   contribution of each mechanism is not isolated here"). It is a stated likely cause, not a
   measured one.

### 0.4 Answering `jiang2024comment` head-on

`jiang2024comment` ("Complex models do not outperform auto-regression" for Hs time-series
prediction) is a live critique of exactly this paper's genre, and it is already in the corpus.
**Engage it explicitly in the Introduction rather than ignoring it.** The response is already
built into the design: a per-frequency-bin ridge autoregressive baseline, rolled out recursively
for the same horizon (`02_methods.tex` §Baselines), included precisely because that critique is
fair for the scalar problem. Because it has no cross-frequency coupling and no non-linearity, it
isolates how much skill comes from genuinely non-linear, cross-bin structure rather than per-bin
temporal extrapolation.

Frame it as: the critique is well-taken for scalar Hs forecasting; the question this paper asks is
whether it still holds when the target is the spectrum, and it is tested rather than assumed.

### 0.5 What is and is not novel — honest boundaries

The overall combination remains unmatched (see `literature/notes.md`, "The gap this corpus does
not fill"). But **three components have precedent, and the manuscript must not overclaim them**.
The first two were turned up by a 2026-09-23 search, the third by a 2026-09-29 search:

- **Partition-conditioned evaluation is not new.** `hanson2009pacific` verified numerical wave
  models per wind-sea/swell partition in 2009. The defensible contribution is narrower: carrying
  an established *numerical-model verification* practice into both the evaluation and the
  **training objective** of a learned forecast, where a single aggregate score remains the norm.
- **The shape/magnitude factorisation is not new in forecasting generally.** `sevlian2018scaling`
  uses "scalar total × normalised shape, multiplied at inference" for electricity load;
  `guo2026loadshape` argues the same divide-and-conquer rationale. No ocean-domain precedent was
  found, so claim the *application* — unit-area spectrum as a decoupled forecast target with the
  physical m₀ = (Hs/4)² relation supplying magnitude — not the factorisation. Note too that
  non-dimensionalising spectra by Hs is routine in wave *characterisation* (JONSWAP-family
  parameterisations), so the normalisation itself is not novel either.
- **Learned forecasting of the full spectrum at a buoy is not new, as post-processing.**
  `filoche2026postprocessing` (2026) maps the ECMWF operational spectral forecast to a buoy's 1D
  spectrum out to 5 days. Claim the *input setting*, not the target: the spectrum forecast from
  the station's own history alone, with no numerical-model input.

  The same paper found that 5 days of buoy history **degraded** its post-processor and was
  rejected by its search (§4.1, §5.2.3). Expect a reviewer to raise this. The answer is in their
  own text: they attribute it to their architecture ("failed to condition the correction on
  recent observations"), and their setting already contains the numerical forecast. Do not
  present their result as showing that buoy history carries no skill, nor dismiss it.

Still clean, uncontested gaps: (1) Wasserstein/optimal-transport distance as a **training-loss**
term for spectral-density prediction, and (2) DMD as an **auxiliary input feature** rather than as
the forecasting method itself.

### 0.6 Why AI/DL at all, rather than only numerical models

Per `personal_notes.md`'s "usefulness of AI/DL models" entry — cite `james2018wave`,
`minuzzi2023lstm`. Numerical spectral wave models (WAVEWATCH III, SWAN) solve the energy balance
equation explicitly, requiring a full forcing chain (atmospheric model output, boundary
conditions, bathymetry) and heavy compute; a trained model instead maps recently observed buoy
spectra directly to a forecast at a fraction of the inference cost, without a coupled
ocean-atmosphere modelling chain. State this as a genuine trade-off, not a one-sided pitch (§5's
"old-vs-new contrast" rule already requires this): the data-driven approach gains speed and can
implicitly learn spectral evolution patterns (swell dispersion, wind-sea growth/decay) without an
explicit physical parameterisation, at the cost of interpretability and the physical guarantees a
numerical model provides by construction. Do **not** lean on an "unavailable at remote/local
sites" framing — neither cited source states that claim; both argue the computational-cost/
surrogate version only.

### 0.7 Scope — state it plainly, it is a design choice not a weakness

- **Per-site models, by design.** The model is trained for a specific location, so no spatial
  (unseen-buoy) generalisation is claimed or required — training per site *is* the approach, not a
  shortfall in it. The temporal hold-out is therefore the right evaluation and should be presented
  without apology. Do not write this up as a limitation.
- **Three buoys with differing climatology — PLANNED, NOT YET DONE.** The study is being extended
  from the single site to three sites chosen for contrasting wave climates. Until those runs
  exist, do not write the Introduction as though multi-site results are in hand; equally, do not
  frame single-site as the paper's final scope. **Flag this to the author whenever drafting text
  whose wording depends on it.**
- **Reported statistics come from seed repetition.** Final results are produced by retraining the
  selected configuration with identical hyperparameters across multiple random seeds, and reported
  as mean ± spread, so the quoted variability reflects training noise rather than tuning quality
  (`02_methods.tex` §Final model).
- **The wind auxiliary channel could not be assessed at the site reported here** (every wind entry
  in its standard meteorological record is a missing-value sentinel); the DMD growth/decay
  features, derived from the spectra themselves, are used instead.

### 0.8 Draft contributions

Tentative, in the author's usual "(i)...; (ii)..." Introduction convention. Refine once Results
exists — **do not present these as final, and do not attach numbers to them yet** (see §0.9):

(i) a transformer with a frequency-structured spectral embedding that forecasts the 1D wave
spectrum autoregressively from a single buoy's own history, predicted in log space so
non-negativity holds by construction; (ii) a shape/magnitude decomposition that separates the
(easier) magnitude problem from the (harder) shape problem, recombined at inference — applying to
wave spectra a factorisation established in other forecasting domains; (iii) a composite spectral
loss (KL-divergence + Wasserstein + soft-max peak height) motivated by per-bin error's inadequacy
for spectral comparison (full justification in `decisions/wasserstein_kl_justification.tex`),
together with a loss-agnostic selection criterion that makes differently-optimised models
comparable; (iv) a peak-resolved, partition-conditioned (wind-sea/swell) evaluation panel that
surfaces multimodal failure behaviour a whole-spectrum metric dilutes away.

**The ablation protocol is methodological, not motivational.** `02_methods.tex` §Incremental
ablation strategy (one change per iteration, matched re-search, seed-spread as the acceptance
threshold, measured cost accounting, and an explicit "candidates tested and not retained" list) is
unusually rigorous and belongs in the paper — but it is *how the work was done*, not *why the work
matters*. Keep it in Methods where it already lives; do not promote it into the Introduction's
motivation or gap statement. At most it earns a sentence in the contributions list or the roadmap
paragraph.

### 0.9 Results are not final — do not quote numbers yet

The analysis is **still in progress**; current results in `results/` are not the final ones. Do not
carry any specific metric value from them into `.tex` prose, into the contributions list, or into
the abstract yet — including the loss-ablation figures already written into `02_methods.tex`
§Incremental ablation strategy, which should be re-checked against the final runs before
submission. Ask the author before quoting any number.

**Planned and flagged for later: a climatological analysis of the three sites.** When it exists it
belongs in the Methods dataset subsection, together with the site characterisation (station
identity, location, wave climate, why these three contrast usefully) — not in the Introduction.

### 0.10 Status of this section

A synthesis, not new source material — every factual claim still needs its own citation check
against `refs.bib` (§3) and its own numeric backing against `results/` (§2) before it lands in
`.tex` prose. Update this section directly if the story changes, rather than letting
`01_introduction.tex` drift away from it.

## 1. Project context

This is a manuscript reporting the transformer-based wave-spectra forecasting work in the
parent repo (`../CLAUDE.md` has the full technical architecture — read it for any modeling
detail, don't guess). Target journal: **Ocean Engineering** (Elsevier), `elsarticle` class,
author-year (Harvard) citation style — see `journal/author_guidelines.md`. Single-buoy study
(NDBC 32012, 2016–2017), forecasting `hs`/`density`/`shape` targets at 12/24/48h lead times.
The manuscript's own framing angle per the journal scope notes: short-term environmental
prediction with a structural/operational hook, not climate science.

Author affiliation: Institute of Mathematics and Statistics, Federal University of Rio
Grande do Sul (UFRGS).

`main.tex` still has open TODOs: title, author list/affiliation, abstract, highlights,
keywords, CRediT, competing interests, funding, data availability, acknowledgements.
`sections/01_introduction.tex`, `03_results.tex`, `04_discussion.tex` are empty stubs.
`02_methods.tex` is substantially drafted but has `[INSERT: Results section reference]`
placeholders still to fill once Results exists.

**Not yet confirmed with the author:** author name(s)/order, exact postal address details
(city/state/country in `main.tex`'s affiliation block), and target submission date. Ask
rather than assume if these matter for a specific edit.

## 2. Source-of-truth rules

**No number, statistic, or figure goes into the manuscript from memory or estimation.**
Every quantitative claim must trace to one of:
- `results/{EXPERIMENT_NAME}/{target}/lead_{N}h/` (per-experiment outputs — check
  `metadata.md` at the experiment root for channel_set/aux_set/architecture)
- `results/RESEARCH_LOG.md` (cross-experiment summary, regenerated by
  `scripts/summarize_results.py`)
- `results/comparisons/` (output of `scripts/compare_versions.py`)
- `results/*ablation*` (output of the loss-ablation pipeline, see
  `scripts/compare_ablation_phases.py` / `evaluate_ablation_phases.py`)

If a number can't be traced to one of these, **say so explicitly and ask** rather than
writing a plausible-sounding value, rounding to "about", or inferring it from an adjacent
number. Same rule for figures: a figure belongs in the manuscript only if it was generated
by `scripts/plot_results.py`, `scripts/plot_cdf_wasserstein.py`, `scripts/plot_ablation_spectra.py`,
or similar, from actual results — never sketched/invented to illustrate a point.

There is currently no `manuscript/figures/` directory — flag this rather than silently
inventing a path when a section needs to reference one.

## 3. Citation rules

**`literature/refs.bib` is the canonical bibliography.** `main.tex` now points
`\bibliography{literature/refs}` at it. The old root-level `references.bib` is stale/unused —
don't add entries there. Both files were empty at the time of writing; `literature/refs.bib`
is where all real entries go from here on.

Rules:
- Only cite works that exist as entries in `literature/refs.bib`, or are documented in
  `literature/notes.md`/`literature/notes/*.md` (Claude-generated per-paper triage records)
  or `literature/personal_notes.md` (the author's own free-form reading notes and ideas —
  read this too before drafting; check it for arguments/framings the author already wants
  used, not just for citations).
- `literature/personal_notes.md` entries are tagged `Reference: <bibkey>`,
  `Reference: <freeform citation, not yet in refs.bib>`, or `Reference: none yet` — see that
  file's own header for how to treat each. A `none yet`/freeform entry backing a claim that
  needs a citation is **not** a substitute for a real `refs.bib` entry — flag it rather than
  citing it directly or inventing BibTeX from the freeform text.
- **Never cite from training-data recall.** If a claim needs a citation not present in
  `literature/`, say so explicitly (e.g. "this claim needs a citation for X — none found in
  literature/, please add one or confirm the source") rather than inventing a plausible
  author/year or pulling one from memory.
- Follow `journal/author_guidelines.md` for citation format (author-year, Elsevier house
  style, `et al.` rules, DOI inclusion, etc.).

Note: `decisions/wasserstein_kl_justification.tex` currently carries its own **self-contained
`\thebibliography`** (Cuturi 2013, Peyré & Cuturi 2019, Kullback & Leibler 1951, etc.) that
has not been merged into `literature/refs.bib`. When drafting Methods/Discussion text that
draws on that decision doc's citations, they need to be added to the real bib file first —
don't assume they're already available to `\cite{}` in the manuscript.

## 4. Decision log usage

Before writing any methodological justification (Methods, Discussion, or a reviewer
response) for a design choice, **check `decisions/` first** for the actual reasoning — don't
reconstruct a plausible-sounding rationale from the code alone. Start with
`decisions/README.md` — it has the full index.

The decision log is two-tiered:
- **`decisions/log/NNN-slug.md`** — short Markdown records of individual ablation/design
  decisions (hypothesis, what changed, before/after evidence with its source cited, keep/
  revert call). This is most of the log — development here was largely ablation-driven
  (implement → run → compare → keep or revert), and most entries are this tier. Several
  entries explicitly flag comparisons as confounded (multiple changes bundled in one study)
  or as open/unresolved trade-offs (e.g. `009-peak-fidelity-objective-metric.md`) — respect
  those caveats; don't cite a flagged number as a clean result.
- **Top-level `decisions/*.tex`** — full standalone LaTeX documents (own
  title/sections/bibliography), reserved for a decision that needs citation-backed prose
  because it's argued in the manuscript itself, not just recorded. Currently one entry:
  `decisions/wasserstein_kl_justification.tex` — full justification for the composite
  spectral loss (RMSE + KL-divergence + Wasserstein-1 + soft-max peak height), including why
  RMSE alone is inadequate for spectral comparison and why each term was added rather than
  replacing RMSE outright. This is the source for that specific loss-design paragraph in
  Methods — pull reasoning from here, not from re-deriving it against `utils/loss.py`.

If a future `.tex` entry appears, treat it the same way (full standalone doc, not a short
note). If a `log/` or `.tex` entry is marked superseded/deprecated/exploratory, treat it as
historical context only — use the current methodology from the code + the newer decision
entry instead, and don't present an exploratory finding as a settled result.

## 5. Style / voice

Calibrated from the author's Methods section in a previous paper (marine-debris
bibliometrics/BERTopic study). Match this register in new prose; `02_methods.tex` already
follows it closely, so use that file as the in-project reference too.

- **British/Commonwealth spelling** throughout: "normalisation", "labelling",
  "generalisation", "minimised", "modelling". Already consistent in `02_methods.tex` — don't
  let edits drift to American spelling.
- **First person plural, active voice** for the authors' own actions/choices ("we used",
  "we applied", "we opted for"), mixed with passive voice for describing the data/process
  itself ("entries without abstracts... were removed", "the dataset includes..."). Never
  first-person singular.
- **Justify every methodological choice inline**, in the same sentence or the next, not just
  state it. Pattern: [choice] + because/to/which [reason], often with a citation backing the
  reason (e.g. "chosen to be reasonably broad", "because its results can be misleading for
  non-spherical clusters"). A design decision stated without its rationale is incomplete by
  this author's standard — check `decisions/` (§4) for the rationale before writing the
  sentence.
- **Numbered sub-steps described in prose**, not bullet lists, when a method has ordered
  stages: "(i) ... ; (ii) ... ; (iii) ...", each later expanded in its own paragraph. Bullets
  are acceptable for parameter/range listings (e.g. hyperparameter search ranges) but not for
  the main procedural narrative.
- **Explicit reproducibility detail**: software name + version + OS/platform when a tool is
  introduced, exact parameter ranges for any search, exact query strings/filters used for data
  collection. Don't summarize a filtering step vaguely — state the concrete rule.
- **Acronyms defined at first use**, spelled out then abbreviated in parentheses, used
  consistently thereafter.
- **Hedged, measured claims** — "is expected to", "It is important to note that", results
  framed against their limitations in the same paragraph they're introduced (e.g. noting what
  a metric does *not* capture right after presenting it). Avoid overclaiming novelty or
  performance.
- **Transitional signposting** between paragraphs: "Finally,", "Nevertheless,", "On the other
  hand,", "However, as in any X task...". Use these to make the argument's structure explicit
  rather than relying on implicit flow.
- Sentence length: moderate-to-long, technical, but not padded — each clause carries
  information (a parameter value, a citation, a rationale), not filler.

**Additional traits from the author's Introduction** (same previous paper) — apply these when
drafting `01_introduction.tex` specifically, on top of the general rules above:

- **High citation density**, especially in the opening paragraphs: nearly every substantive
  claim carries its own citation, and stakeholder/impact lists attribute a separate citation
  to each item rather than one citation covering the whole list (e.g. "scholars,
  environmental regulation agencies (X 2020; Y 2022), policymakers (Z 2016) and the general
  population (W 2022)"). Grouped citations are semicolon-separated and chronological/
  alphabetical, consistent with `journal/author_guidelines.md`.
- **Explicit gap statement**, stated plainly rather than implied: "To our knowledge, no
  studies have yet attempted to..." / "no comprehensive ... study exists." This claim must
  itself be defensible from what's actually in `literature/` — don't assert a gap without
  having checked the literature folder for a counterexample first (§3, §8).
- **Old-vs-new method contrast with an honest trade-off**, not a one-sided pitch: "By
  contrast, X leverages Y..., thereby improving Z, even if at the expense of W." Carries the
  same anti-overclaiming discipline from the Methods rules into the Introduction's framing of
  the model itself (e.g. this project's transformer approach vs. a simpler baseline).
- **Contributions stated as an explicit numbered list** near the end of the introduction —
  "(i) ...; (ii) ...; (iii) ...; (iv) ..." — mirroring the same prose-numbered-list device used
  for procedural steps in Methods.
- **Closing roadmap paragraph**: "The remainder of this paper is organised as follows.
  Section 2 presents... Section 3 reports..." naming each section and, where a section has
  sub-parts, listing them the same way.
- **Rounded totals with an exact breakdown in parentheses** when reporting corpus/sample
  counts, e.g. "over 12,000 publications (roughly 9,800 articles, 1,400 reviews, ...)" — round
  the headline number but disclose the real composition immediately after. Any such number in
  this manuscript still needs a §2 source — the disclosure style doesn't relax the
  traceability rule, it's just how the number gets presented once traced.

This section is calibrated, not provisional — treat it as settled unless the author revises it.

## 6. Write the science, not the repository

The manuscript describes a forecasting *method*, not this codebase. A reader must never be
able to tell a Python implementation sits behind it. This has been a recurring failure mode —
correct it actively, don't just avoid it passively.

**Never let repo/code framing leak into prose:**
- No code identifiers, class/function names, file names, or repo vocabulary
  (`FreqDimEmbedding`, `prepare_X`, `channel_set`/`aux_set`, `EXPERIMENT_NAME`, "the encoder
  pathway bypasses...", etc.). If a concept only makes sense by naming its class, translate it
  into the underlying statistical/physical operation instead.
- No implementation-detail mentions — array/tensor shapes, indices, "numpy array", "broadcast
  across the window", memory/compute bookkeeping (`$185\,\mu$s per sample`-style asides)
  unless the number itself is a genuine, citable result the paper is reporting (e.g. inference
  latency as a stated contribution) rather than incidental engineering trivia.
- **Translate by function, not by repo role.** Example: the DMD-derived scalars are an
  *auxiliary input channel* only in the code's data-fusion sense — in the paper they must be
  introduced as what they scientifically are (features characterising the growth/decay
  dynamics of the recently observed spectral time series via Dynamic Mode Decomposition),
  justified on physical/statistical grounds, not on "how they're fused into the encoder."
  Section headings like "Auxiliary feature fusion" describe a code pathway, not a modelling
  choice — retitle around what the feature *is* and why it's included.

**Never narrate the project's own development history in the main text:**
- Phrases like "unlike in the earlier configuration of this work", "was superseded by",
  "reported for comparability with the earlier configuration... in which it was the selection
  criterion", "candidates did not survive this procedure, and are reported because..." read as
  a changelog, not a methods section. The manuscript states and justifies the *final* chosen
  method; it does not narrate the sequence of things tried before it.
- This is not the same as reporting a real ablation study. A deliberate ablation that supports
  a claim belongs in Results, framed as an experiment ("we compared configuration A against B
  to test X"), with its own evidence — not narrated as "the project's history" in Methods.
  Internal engineering trial-and-error (what was tried, kept, or reverted, and why) belongs in
  `decisions/` (§4), not in the manuscript, unless it rises to a reported ablation.

Before writing or revising a Methods/Results passage, check whether a sentence would still
make sense to a reader who has never seen this repository. If it wouldn't, rewrite it in
domain terms (oceanography, statistics, ML methodology) rather than repo terms — don't just
soften the wording while keeping the code-shaped framing.

## 7. Author guidelines

`journal/author_guidelines.md` is the authority on formatting, word/page limits, section
order, citation style, figure/table requirements, and submission checklist for Ocean
Engineering. `journal/elsarticle/` has the actual LaTeX class and reference templates
(`elsarticle-template-harv.tex` is the one `main.tex` is based on).

**Check `journal/author_guidelines.md` before restructuring or reformatting any section** —
e.g. before reordering sections, changing citation style, adjusting abstract/highlights
length, or altering figure/table conventions. Don't assume a generic journal convention
applies here.

## 8. Working process

- Edit one section at a time. Don't rewrite the whole manuscript in a single pass.
- Treat prose edits like code changes: propose them as a clear, reviewable diff (show
  old → new, or use Edit with clear old_string/new_string) rather than silently rewriting
  large blocks of existing text.
- Ask before deleting or majorly restructuring existing text — including TODO placeholders
  in `main.tex`, which mark real open decisions, not junk to clean up.

## 9. Always flag to the author

- Any numeric/statistical claim or figure that can't be traced to a specific file under
  `results/` (see §2).
- Any citation needed but not present in `literature/refs.bib` (see §3).
- Any place where a methodological rationale had to be inferred rather than found in
  `decisions/` or grounded in `literature/`.
- Any `[INSERT: ...]` placeholder in the existing `.tex` files being filled in — confirm the
  target section/number is correct rather than resolving it silently.
- Any existing sentence you notice that violates §6 (code/repo leakage or project-history
  narration) while working on something else nearby — point it out even if it's outside the
  edit you were asked to make.
