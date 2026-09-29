# Personal notes

Free-form, author-written notes about related work — ideas, half-formed connections, framings to
use in the manuscript, quotes or claims worth checking — kept separate from the per-paper triage
records in `notes.md` / `notes/*.md`, which are Claude-generated summaries produced by a formal
triage pass over PDFs already vetted into `refs.bib`. This file is the author's own working notes
and is not itself produced or verified by that triage process; some entries will have a solid
reference, some won't yet.

## How to use this file (for Claude)

- Every note here is authorial intent/knowledge — don't re-verify it the way you'd treat a web
  result, but *do* still check whether its reference actually resolves to a `refs.bib` entry
  before citing it in the manuscript.
- Each entry is tagged with one of:
  - `Reference: <bibkey>` — already in `literature/refs.bib`, safe to `\cite{}` directly.
  - `Reference: <freeform citation the author has — author/year/title/DOI, etc.>` — not yet in
    `refs.bib`. A `refs.bib` entry needs to be added (and metadata confirmed, per the sourcing
    rules in `../CLAUDE.md` §3) before this can be cited — flag it rather than inventing the
    BibTeX entry from the freeform text alone.
  - `Reference: none yet` — an idea, observation, or framing with no source attached. Fine to use
    as the author's own reasoning/argument in prose that doesn't require a citation; if the
    surrounding claim *does* need one, say so explicitly rather than pulling a plausible citation
    from memory or from an unrelated entry in `literature/notes/`.
- If an entry's freeform reference looks like it might already match something in
  `literature/notes/` or `literature/refs.bib`, say so and propose consolidating rather than
  creating a duplicate bib entry.

## Entries
### weather - numerical vs AI models

Reference: rasp2024weatherbench2; benbouallegue2024rise — sourced 2026-09-23, superseding the
freeform links below (https://doi.org/10.48550/arXiv.2308.15560 = WeatherBench 2, which has a
peer-reviewed JAMES version; https://owb.brightband.com/?start=2026-08-20&end=2026-09-19)

AI/DL models, especially transformer architectures, consistently outperform numerical modelling for meteorological forecasting. For ocean forecasting, the improvements have been slower.

**CORRECTION (2026-09-23) — "consistently outperform" is not supportable as written.** The
references now exist, but neither licenses that verb, and this needs fixing before the claim
reaches the manuscript:

- `rasp2024weatherbench2` (WeatherBench 2) is deliberately neutral — it presents results for both
  physical and data-driven models and explicitly discusses caveats in the evaluation setup.
- `benbouallegue2024rise` is ECMWF's own operational-like head-to-head of PanguWeather against
  the IFS, and reports "comparable accuracy for both global metrics and extreme events" — plus
  named drawbacks: overly smooth forecasts, bias growing with lead time, poor tropical-cyclone
  intensity.

Recommended wording: data-driven models now **match or approach** operational NWP skill, rather
than "consistently outperform". If the stronger verb is wanted, the sharp claim lives in GraphCast
(Lam et al. 2023, *Science* 382(6677), 1416-1421, doi 10.1126/science.adi2336 — Crossref-verified
but **not yet in refs.bib and no PDF**; file it properly first if it gets used).

Second caveat: both references are **atmosphere-only**. They say nothing about ocean or wave
forecasting, so the "improvements have been slower for ocean forecasting" half of this note is
still uncited. Either hedge it as the authors' own observation or source it separately.

**Unexpected bonus — the more valuable use of `benbouallegue2024rise`.** It independently
documents **over-smoothing as a characteristic failure mode of data-driven forecast models**,
verified operationally, in a different geophysical domain. That is exactly the failure mode this
project's composite loss and peak-resolved evaluation exist to counter. It reframes our central
problem from "our wave model blurs peaks" into "a documented, cross-domain pathology of learned
forecasters, which we measure and correct" — a much stronger position. Use it in the
over-smoothing paragraph, not just the opening one.

### ocean forecasts — ML/AI focus on scalar params, not the spectrum

Reference: makarynskyy2004

Also see zhou2021emdlstm.

Developments in ocean forecasting using statistical/AI models have focused on scalar bulk
parameters (chiefly Hs), not the full directional spectrum. Lineage: early ANN forecast
correction (makarynskyy2004, Hs/Tz at 1-24h) through EMD-LSTM (zhou2021emdlstm, Hs at 3-72h) to
the most recent transformer work (kim2026metoformer, joint scalar metocean variables, not the
spectrum) — the pattern holds from 2004 through 2026, which is what makes it a real gap rather
than an artifact of older literature. Pairs with the "weather - numerical vs AI models" note
above: AI's edge over numerical models has been slower to show in ocean forecasting generally,
and within ocean forecasting specifically it has concentrated on scalar outputs rather than
spectral ones — this project's spectrum-forecasting target sits at the harder, less-explored end
of both gaps at once.

### optimization metric gap
tied to "ocean forecasts — ML/AI focus on scalar params, not the spectrum" note above, we noticed that the metrics used for optimization and evaluation of scalar parameters don't extrapolate well to 1D spectra. that argument should be linked to our metrics ablation. additionally, we needed a comparable metric between them, since a model optimized for rmse, would naturally perform better in rmse than a model optimized for w2, but neither would generelize well for the full spectrum.

### usefulness of AI/DL models

Reference: james2018wave; minuzzi2023lstm — see also gao2025jgr (already in corpus)

Why forecast wave spectra with AI/DL instead of relying purely on numerical (physics-based)
wave models?

- Numerical spectral wave models (e.g. WAVEWATCH III, SWAN) solve the energy balance equation
  explicitly, which requires a full forcing chain — atmospheric model output, boundary
  conditions, bathymetry — and is computationally expensive to run and maintain at high
  resolution/update frequency. This is a well-established, citable motivation: james2018wave
  reports a trained ML surrogate reproducing SWAN's Hs/period output at "a fraction (< 1/1,000th)
  of the computation time," stating plainly that "computational expense is often a major
  limitation of real-time forecasting systems"; minuzzi2023lstm (UFRGS — this manuscript's own
  institution) poses the same question directly for buoy-based Hs forecasting, asking whether
  data-driven models can "act as a physical model surrogate, with computational time and
  accuracy that are superior to" WAVEWATCH III/SWAN. A trained AI/DL model instead maps recently
  observed buoy spectra (plus cheap aux data like wind) directly to a forecast, at a fraction of
  the inference cost and without needing a coupled ocean-atmosphere modelling chain. This is
  attractive specifically for local, buoy-scale, short-lead forecasting, where a site-specific
  high-resolution numerical run may not be available or timely — though note neither source
  states that "unavailability at remote/local sites" framing explicitly; both argue the
  computational-cost/surrogate version of this point, which is the safer claim to attribute to
  them directly.
- Ties to the "weather - numerical vs AI models" note above: AI/DL (transformers especially)
  already demonstrably outperforms NWP for meteorological forecasting; ocean forecasting has
  been slower to show the same gains. That gap is itself a motivation for pushing AI methods
  further here, rather than assuming the meteorological result doesn't transfer to ocean
  forecasting.
- Ties to the "ocean forecasts — ML/AI focus on scalar params, not the spectrum" note above:
  existing AI/ML ocean-forecasting work has stopped at scalar bulk parameters (Hs, Tz).
  Forecasting the full directional spectrum is the harder problem within this class, but it's
  the physically richer target — any bulk parameter (Hs, Tm02, directional spread) is
  derivable from the spectrum, but not the reverse. A model that forecasts the spectrum is
  therefore strictly more useful downstream (structural design loads, directional-dependent
  operations, wind-sea/swell partitioning) than one that only forecasts Hs.
- Data-driven models can also implicitly learn spectral evolution patterns (swell dispersion,
  wind-sea growth/decay) directly from historical buoy records, without an explicit physical
  parameterisation of those processes. Worth stating in the manuscript as a trade-off, not
  just an advantage: it comes at the cost of interpretability and the physical guarantees a
  numerical model provides by construction.

Still needs a reference before use: the "site-specific numerical run may not be available or
timely" framing specifically (last clause of the first bullet) — james2018wave/minuzzi2023lstm
cover the computational-cost/surrogate version of the argument but not this exact framing; flag
if the manuscript prose leans on it directly rather than the hedged version above.

### why the spectrum and not Hs — the consequence argument

Reference: orimoloye2019bimodal; simao2025bimodal (the latter is UNREAD — metadata only, see its note)

The scalar-vs-spectrum gap note above is an *information* argument (the spectrum is richer; bulk
parameters are derivable from it but not the reverse). On its own that is a weak motivation — a
reviewer can fairly ask "richer for what?". The Introduction needs the *consequence* version
alongside it, which is the one an Ocean Engineering readership actually cares about: the response
of a vessel or moored structure is frequency-dependent, so two sea states carrying identical
total energy — identical Hs — produce materially different motions, loads and fatigue if that
energy is distributed differently in frequency. Hs is therefore not a sufficient statistic for
the thing the forecast is ultimately used to predict.

orimoloye2019bimodal is the clean support for the premise: it constructs **energy-conserved**
bimodal sea states (total energy held constant, swell percentage and swell peak period varied)
and finds the wave height distribution's kurtosis and skewness varying with them — same energy,
different behaviour. Its own motivation sentence is quotable for us: "An understanding of the
wave height distribution of a sea state is important in forecasting extreme wave height and
lifetime fatigue predictions of marine structures."

Careful with the direction of its result: it reports *greater* non-linearity in unimodal than in
bimodal seas at matched energy. So the citable claim is "spectral composition changes behaviour
at fixed Hs", NOT "bimodal seas are more severe". Don't let the prose drift into the second.

simao2025bimodal would carry this one step further — from wave statistics to the extreme response
of real moored systems, offshore Brazil — which is exactly the step needed. But its abstract was
not retrievable (ASME paywall, HTTP 403), so nothing may be attributed to it yet. Either get the
PDF or cite it only for the existence of the research question.

This same pair also motivates the multimodal/partition-conditioned evaluation panel: bimodality is
not a curiosity to be handled for completeness, it is precisely the regime where bulk parameters
are least informative — which is why an evaluation that only reports whole-spectrum aggregates is
measuring the wrong thing there.

### spectral verification practice — what standard is, and where we depart

Reference: ecmwf2019lcwfvparameters; ecmwf2026lcwfvproject; hernandez2025intercomparison; hanson2009pacific
(bidlot2002intercomparison retained for historical origin only)

**Update 2026-09-24: step 1 is now sourced from current practice, not 2002.** The Introduction
cites three sources for step 1:
- `hernandez2025intercomparison` for scale: an ongoing WMO Lead Centre at ECMWF, comparing 18
  systems. It does not name the variables.
- `ecmwf2026lcwfvproject`, the author-supplied Project page, for verification against buoy
  observations.
- `ecmwf2019lcwfvparameters`, the Lead Centre's Parameters page, for the variables: "6 parameters
  were agreed for common verification". These are Hs, peak period, mean zero-crossing period, mean
  direction and 10-m U/V wind, "based on the full 2-D spectrum". The spectrum itself is not among
  them.

The Project page does not list the parameters itself, which is why the Parameters page is cited
alongside it. The Lead Centre's Verification results page also states that its QC procedure
originates in Bidlot et al. (2002), so the lineage in step 1 is sourced after all.

Sourced 2026-09-23 to put a citation behind the claim our Methods currently argues from first
principles: that whole-spectrum aggregate error hides errors confined to a narrow peak.

The honest three-step story for the Introduction:
1. **Standard operational practice is bulk-parameter verification.** `bidlot2002intercomparison`
   is the founding intercomparison behind what became the WMO Lead Centre for Wave Forecast
   Verification — five centres verifying against buoys on Hs, peak period and wind speed only.
2. **The wave-modelling community already published the critique.** `hanson2009pacific` verifies
   three numerical models *per wind-sea/swell partition* and opens with our exact premise:
   "Although mean or integral properties of wave spectra are typically used to evaluate numerical
   wave model performance, one must look into the spectral details to identify sources of model
   deficiencies." Its introduction is blunter — bulk parameters "can mask higher-order
   deficiencies".
3. **Machine-learning wave forecasting inherited the habit but not the critique.** That is our
   actual gap.

**Important consequence — do not claim we invented partition-conditioned evaluation.** It has been
established practice for numerical wave models since 2009, well before `rogers2025espc`. Our
defensible contribution is carrying it across to a *learned* forecast, and further, into the
**training objective** rather than only the verification step. Stated that way it is still a real
contribution and it is now citation-backed instead of asserted.

### shape/magnitude split — precedent exists, outside oceanography

Reference: sevlian2018scaling; guo2026loadshape

A dedicated search (2026-09-23) for precedent on the E(f,t) = S(f,t)·m₀(t) decomposition found
**no ocean-domain precedent**, but a clear one in short-term electricity load forecasting.
`sevlian2018scaling` forecasts daily total consumption and a normalised daily shape separately
and multiplies them at inference — structurally identical to what we do. `guo2026loadshape` makes
the same divide-and-conquer *argument* we make: one model is peak-optimal, another shape-optimal,
so decouple them.

So: don't claim the factorisation as an invention. Claim the **application** — the unit-area
spectrum as a decoupled forecast target, with the physical m₀ = (Hs/4)² supplying magnitude, and
the specific motivation of countering spectral peak under-prediction. Also note that
non-dimensionalising a spectrum by Hs is routine in wave *characterisation* (JONSWAP-family
parameterisations), so the normalisation itself isn't novel either — only its use as a separately
forecast target.

That the same reasoning arose independently in an unrelated domain is worth one sentence in the
Discussion as support for the design, not a threat to it.

### pre-norm transformer — citing the norm_first arrangement

Reference: xiong2020prenorm

This is the citation for "the model is an encoder-decoder Transformer in the pre-normalisation
arrangement" in Methods § Model architecture. It is already `\citep`'d there. It is the standard
analysis of *why* LayerNorm placement matters. It is not the origin of Pre-LN, which the paper
credits to Baevski & Auli 2018, Child et al. 2019 and Wang et al. 2019.

What it licenses: moving LayerNorm inside the residual branch keeps gradients well-behaved
across layers at initialisation, and makes training less sensitive to the learning-rate
schedule. What it does not license: "more stable at the deeper end of the search space". That
depth argument is Wang et al. 2019's, and our search space is only 1-4 layers per side anyway.
Reword that clause rather than hang it on this paper.

Watch for the obvious reviewer question. Xiong's headline is that Pre-LN makes LR warm-up
unnecessary, yet we use a 5-epoch warm-up (decision 022). Our defence is that 022 addresses
instability across a very wide *sampled* LR range, not a single tuned rate. Xiong shows that
warm-up *can* be dropped, not that it must be.

<!--
New entry template — copy this block:

### <short topic slug>

Reference: <bibkey | freeform citation info | none yet>

<freeform notes — argument, quote, connection to this project, open question, etc.>
-->
