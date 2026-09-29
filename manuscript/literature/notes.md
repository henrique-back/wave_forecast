# Literature notes

Per-paper triage records for everything in `literature/`, so a future session (human or Claude)
never has to re-open a PDF just to remember what it says or whether it's worth citing. Each
paper added to this folder gets one record in `notes/<bibkey>.md` (citation, summary, relevance
tier, suggested manuscript use) and a matching entry in `literature/refs.bib` if its metadata was
independently confirmed. See `../CLAUDE.md` §3/§4 for how citations and the decision log are
meant to be used together — this file plays the same role for literature that `decisions/README.md`
plays for design decisions.

Triage performed 2026-09-18 (5 parallel PDF-reading passes + 3 websearch verification passes).
Every fact below traces to either the paper's own PDF or a live search result found that day —
none are from training-data recall (see `../CLAUDE.md` §3).

## Index

| Key | Title (short) | Year | Relevance | In `refs.bib`? |
|---|---|---|---|---|
| [makarynskyy2004](notes/makarynskyy2004.md) | ANN-corrected numerical wave forecasts | 2004 | High | Yes |
| [fisher2021sparse](notes/fisher2021sparse.md) | Sparse buoy array, phase-resolved prediction | 2021 | Medium | Yes |
| [hwang2024decnn](notes/hwang2024decnn.md) | DeCNN spectrum reconstruction, KL loss | 2024 | High | Yes |
| [sun2025gpgnn](notes/sun2025gpgnn.md) | G-PGNN physics-guided generative NN | 2025 | High | Yes |
| [kim2026metoformer](notes/kim2026metoformer.md) | MetoFormer-HQ, PatchTST buoy metocean transformer | 2026 | High | Yes |
| [ding2023eofeemd](notes/ding2023eofeemd.md) | EOF-EEMD-SCINet regional Hs/Tp | 2023 | Medium | Yes |
| [kwon2026transformer](notes/kwon2026transformer.md) | Transformer ensemble, FPSO directional-spectrum params | 2026 | High | Yes |
| [cao2025bohai](notes/cao2025bohai.md) | Encoder-decoder regional spectra, Bohai Sea | 2025 | Medium-High | Yes |
| [mandal2005bpnn](notes/mandal2005bpnn.md) | Small BP-NN, scalar bulk params only | 2005 | Low | Yes |
| [sakhare2009svr](notes/sakhare2009svr.md) | SVR/model trees, spectral shape from Hs/Tz | 2009 | Medium | Yes |
| [nielsen2023shipbuoy](notes/nielsen2023shipbuoy.md) | Ship-as-wave-buoy, CNN+physics inversion | 2023 | Medium | Yes |
| [naithani2005ann](notes/naithani2005ann.md) | ANN spectral shape from bulk params | 2005 | High | Yes |
| [jiang2024comment](notes/jiang2024comment.md) | "Complex models do not outperform auto-regression" | 2024 | Medium-High | Yes |
| [liu2024markov](notes/liu2024markov.md) | Markov chain, synthetic sea-level, seconds-scale | 2024 | Low | Yes |
| [zeng2020seq2seq](notes/zeng2020seq2seq.md) | Seq2seq+attention spectrum from acceleration | 2020 | High | Yes |
| [gao2025jgr](notes/gao2025jgr.md) | Physics-guided DL, wind→directional spectra | 2025 | Medium | Yes |
| [portilla2009](notes/portilla2009.md) | Peak partitioning, wind-sea/swell identification | 2009 | High | Yes |
| [zhou2021emdlstm](notes/zhou2021emdlstm.md) | EMD-LSTM Hs forecasting, NDBC buoys | 2021 | High | Yes |
| [kwon2025jmse](notes/kwon2025jmse.md) | FPSO motion → spectrum-param inverse estimation | 2025 | Low | Yes |
| [liu2025cnnxlstm](notes/liu2025cnnxlstm.md) | CNN+xLSTM regional spectra, freq-weighted loss | 2025 | High | Yes |
| [namekar2006ann](notes/namekar2006ann.md) | ANN Hs/Tz→spectrum, peak-underestimation bias | 2006 | High | Yes |
| [kambekar2009asce](notes/kambekar2009asce.md) | GP/SVR/model trees Hs/Tz hindcast | 2009 | Medium | Yes |
| [song2023jpo](notes/song2023jpo.md) | CNN, wind→directional spectra (JPO) | 2023 | High | Yes |
| [cooper2023thesis](notes/cooper2023thesis.md) | Bures-Wasserstein metric learning (unrelated) | 2023 | Low | Yes (unlikely to be cited) |
| [elkhrachy2023water](notes/elkhrachy2023water.md) | EANN/WANN Hs+spectrum forecasting | 2023 | High | Yes |
| [ardhuin2025waves](notes/ardhuin2025waves.md) | Wave-physics course notes (background only) | 2025 | Medium | Yes |

### Web-only comparators — no local PDF, found 2026-09-21 (find-reference)

Found via `/find-reference` (2026-09-21) while sourcing the two flagged claims in
`personal_notes.md`'s "usefulness of AI/DL models" entry: numerical-model computational cost,
and the buoy-based/local-forecasting argument. Same treatment as the other web-only entries
below — metadata independently confirmed via Crossref/arXiv, no local PDF retrieved.

| Key | Title (short) | Year | Relevance | In `refs.bib`? |
|---|---|---|---|---|
| [james2018wave](notes/james2018wave.md) | ML surrogate for SWAN, <1/1000th compute time | 2018 | High | Yes |
| [minuzzi2023lstm](notes/minuzzi2023lstm.md) | LSTM Hs forecast, 7 Brazilian buoys, UFRGS — **local PDF added 2026-09-29, read in full** | 2023 | High | Yes |

**Update 2026-09-29:** the published PDF of `minuzzi2023lstm` is now in `literature/`, and its
note was rewritten from a full read. Before citing, check two corrections to the earlier version
of the note. First, most results are scored against ERA5 rather than buoy observations, and
"accuracy" means 100% − MAPE, with no persistence baseline. Second, its claim to improve on a
physical model is an inference from ERA5, not a head-to-head comparison.

### Web-only comparators — no local PDF, found 2026-09-23 (find-reference)

Found via `/find-reference` (2026-09-23) while sourcing the Introduction's operational premise —
that significant wave height alone does not determine a sea state's engineering consequences,
because structural and vessel response depends on *where in frequency* the energy sits. Metadata
independently confirmed via Crossref (`api.crossref.org`). Neither PDF could be retrieved
automatically: `orimoloye2019bimodal` is gold OA (CC-BY) but MDPI blocks automated download
(fetch manually from its DOI), and `simao2025bimodal` is paywalled on the ASME Digital
Collection (HTTP 403).

**Verification depth differs between the two, and this matters:** `orimoloye2019bimodal` is
verified at *abstract* level (its key sentences are quoted in its note), whereas
`simao2025bimodal` is verified at *metadata* level only — its abstract could not be retrieved at
all, so no finding may be attributed to it until someone reads it. See each note.

| Key | Title (short) | Year | Relevance | In `refs.bib`? |
|---|---|---|---|---|
| [orimoloye2019bimodal](notes/orimoloye2019bimodal.md) | Equal-energy bimodal seas, wave-height distribution | 2019 | High | Yes |
| [simao2025bimodal](notes/simao2025bimodal.md) | Bimodal seas → mooring extreme response, offshore Brazil | 2025 | Medium-High | Yes (unread — see note) |

### Local PDF obtained 2026-09-23 (find-reference, Introduction framing)

Four further references sourced the same day for the Introduction's three remaining framing
claims — the data-driven-vs-NWP state of play, standard spectral verification practice, and
precedent for the shape/magnitude decomposition. All Crossref-verified; PDFs are in
`literature/`. **Read the version caveats in each note before quoting**: three of the PDFs are
arXiv preprints of the published article cited in `refs.bib`, and the Bidlot PDF is partial.

| Key | Title (short) | Year | Relevance | In `refs.bib`? |
|---|---|---|---|---|
| [rasp2024weatherbench2](notes/rasp2024weatherbench2.md) | WeatherBench 2, data-driven vs physical NWP benchmark | 2024 | Medium-High | Yes |
| [benbouallegue2024rise](notes/benbouallegue2024rise.md) | ECMWF assessment of ML weather forecasts; over-smoothing | 2024 | High | Yes |
| [hanson2009pacific](notes/hanson2009pacific.md) | Partition-based verification of 3 numerical wave models | 2009 | High | Yes |
| [bidlot2002intercomparison](notes/bidlot2002intercomparison.md) | Operational wave verification on bulk params, 5 centres | 2002 | Medium-High | Yes |
| [hernandez2025intercomparison](notes/hernandez2025intercomparison.md) | Ocean-model intercomparison review; WMO wave Lead Centre, 18 systems (added 2026-09-24) | 2025 | Medium | Yes |
| [ecmwf2019lcwfvparameters](notes/ecmwf2019lcwfvparameters.md) | WMO LC-WFV: the 6 bulk parameters verified, spectrum not exchanged (web page, added 2026-09-24) | 2019 | Medium-High | Yes (web) |
| [ecmwf2026lcwfvproject](notes/ecmwf2026lcwfvproject.md) | WMO LC-WFV objectives: verification against buoy data, 18 centres (web page, added 2026-09-24) | 2026 | Medium | Yes (web) |

**Update 2026-09-24:** `bidlot2002intercomparison` is no longer cited in the Introduction. The
two LC-WFV pages above replace it as the current, primary source for bulk-parameter-only
verification. It stays in the corpus, because the Lead Centre's own "Verification results" page
cites it as the origin of its quality-control procedure.
| [sevlian2018scaling](notes/sevlian2018scaling.md) | Total × normalised-shape load forecast (precedent) | 2018 | Medium | Yes |
| [guo2026loadshape](notes/guo2026loadshape.md) | Peak × normalised-curve load forecast (precedent) | 2026 | Medium | Yes |

**Two of these narrow previously-claimed gaps — see "The gap this corpus does not fill" below,
updated accordingly.** `hanson2009pacific` shows partition-conditioned verification is
established practice for numerical wave models (since 2009), and `sevlian2018scaling` /
`guo2026loadshape` show the shape/magnitude factorisation is established in load forecasting.
Neither kills the manuscript's contribution, but both change how it must be worded.

### Local PDF obtained 2026-09-29 (find-reference, Methods — architecture)

A standard method citation for the pre-normalisation (`norm_first=True`) arrangement, already
`\citep`'d in `02_methods.tex` § Model architecture. Metadata confirmed via the PMLR landing
page and the arXiv API. The PDF is the published PMLR version. **Read its note before widening
its use.** It supports the gradient-at-initialisation / warm-up-sensitivity argument, *not* the
"more stable at the deeper end of the search space" wording currently in Methods. It also sits
awkwardly beside the model's own LR warm-up (decision 022).

| Key | Title (short) | Year | Relevance | In `refs.bib`? |
|---|---|---|---|---|
| [xiong2020prenorm](notes/xiong2020prenorm.md) | Pre-LN vs Post-LN Transformer, gradients & warm-up | 2020 | Medium | Yes (no DOI — PMLR) |

### Local PDF obtained 2026-09-29 (Methods — spectral partitioning)

The author supplied it as the missing `violantecarvalho2009` source. It turned out to be the 2002
JOMAE paper, and it does **not** state the γ\* criterion (see its note, and "Open issues" below).
Metadata was confirmed via Crossref and the PDF's own title page. The PDF is the complete
published ASME version.

| Key | Title (short) | Year | Relevance | In `refs.bib`? |
|---|---|---|---|---|
| [violantecarvalho2002](notes/violantecarvalho2002.md) | JONSWAP-fit partitioning, wind-sea growth under swell, Campos Basin (γ\* precursor, not source) | 2002 | Medium | Yes |

### Web-only — no local PDF, found 2026-09-29 (Introduction, buoy directional estimation)

`\citep{gorman2018}` was already in `01_introduction.tex` with no `refs.bib` entry and no corpus
record, so it was most likely written from recall. A search identified the matching paper, and
Crossref confirmed the metadata. It is open access (CC BY-NC-ND), but ScienceDirect blocks
automated download, so it is **verified at abstract level only**. Its note checks the
Introduction sentence clause by clause. The "fail in precisely the multi-modal states" clause
goes beyond the abstract, which only says existing estimators cannot handle *more than two
directional* peaks.

| Key | Title (short) | Year | Relevance | In `refs.bib`? |
|---|---|---|---|---|
| [gorman2018](notes/gorman2018.md) | Buoy directional spectra beyond the "first five" moments; >2 directional peaks | 2018 | Medium-High | Yes (abstract-level only — see note) |

### Web-only — no local PDF, found 2026-09-29 (find-reference, physical-model comparison)

Found during the 2026-09-29 search for published verification of numerical-model spectral
forecasts, to compare the manuscript's forecasts against. Crossref confirmed the metadata and
supplied the abstract. The article is open access (CC BY-NC 4.0), but Wiley blocks automated
download, so it is **verified at abstract level**. The lead-time error table in its note comes from
the authors' Zenodo code snapshot (doi:10.5281/zenodo.22557273). That snapshot predates the
published text, and its ML numbers differ slightly from the abstract's.

Two points in the note matter most. First, it **narrows the Introduction's layer-2 gap**: it is a
learned 1D-spectrum forecast at a buoy, as post-processing of the ECMWF forecast (see "The gap this
corpus does not fill" below). Second, its finding that buoy history was *not* selected as an input
comes from the snapshot docs only.

| Key | Title (short) | Year | Relevance | In `refs.bib`? |
|---|---|---|---|---|
| [filoche2026postprocessing](notes/filoche2026postprocessing.md) | DL post-processing of the ECMWF spectral forecast to a buoy's 1D spectrum, 5-day leads, NW Australia | 2026 | High | Yes (abstract-level only — see note) |

### Web-only comparators — no local PDF

Found via the gap-verification websearch (2026-09-18), not supplied by the author. Given the same
per-paper note treatment as the local corpus above, but each note's `source_file` says so
explicitly and flags that the citation details come from search snippets rather than a
PDF-in-hand read — re-verify before quoting beyond what's in the note. See "Gap-verification
comparators" below for what each is closest-analog to.

| Key | Title (short) | Year | Relevance | In `refs.bib`? |
|---|---|---|---|---|
| [meng2023windswell](notes/meng2023windswell.md) | Wind-sea/swell separation by deep learning | 2023 | Medium-High | Yes (no DOI) |
| [breunung2023dmd](notes/breunung2023dmd.md) | DMD as the forecasting mechanism itself | 2023 | Medium | Yes (no DOI) |
| [rogers2025espc](notes/rogers2025espc.md) | Navy ESPC partition-conditioned skill assessment | 2025 | Medium | Yes (no DOI) |

Standard method/tooling and wave-physics-background citations (no local PDF, all confirmed via
websearch 2026-09-18): `vaswani2017attention`, `loshchilov2019adamw`,
`bengio2015scheduledsampling`, `akiba2019optuna`, `bergstra2011tpe`, `kullback1951`,
`schmid2010dmd`, `peyre2019`, `cuturi2013sinkhorn`, `longuethiggins1963`, `kuik1988`,
`ndbc2003techdoc`, `holthuijsen2007`.

## The gap this corpus does not fill

No paper found — locally or via websearch — does the specific combination this manuscript
reports: a transformer forecasting the full 5-channel directional spectrum autoregressively from
a single buoy's own history, with a shape/magnitude split, a KL+Wasserstein+peak-height composite
loss, DMD auxiliary dynamical features, and a partition-conditioned (wind-sea/swell) evaluation
panel. Closest analogs and where each falls short:

- **kim2026metoformer** — transformer, multi-horizon, buoy-based — but joint scalar metocean
  variables (not the full directional spectrum), patch-based (not frequency-structured per-bin)
  embedding.
- **kwon2026transformer** — transformer applied to directional-spectrum parameters — but from
  FPSO motion inversion, not buoy-history forecasting.
- **song2023jpo** / **gao2025jgr** — CNN, directional spectra — but from wind fields, not
  spectral autoregression, and targets are model-simulated rather than observed buoy spectra.
- **naithani2005ann** / **sakhare2009svr** — shape decoupled from bulk params — but the reverse
  direction (params→shape via classical/ANN regression), not a learned forecast pair recombined
  at inference.
- **filoche2026postprocessing** (added 2026-09-29) — a learned forecast of the buoy-measured 1D
  spectrum, from 3 h out to 5 days, with an attention-based encoder-decoder. It is the closest
  analog found for "ML forecasting the spectrum at a buoy". But it post-processes the ECMWF
  operational spectral forecast, with the numerical-model spectrum as input, rather than
  forecasting from the buoy's own history. Its swell/wind-sea split is a fixed 0.10 Hz cut.

### One further narrowing, 2026-09-29 — the layer-2 "none forecast the spectrum" claim

`filoche2026postprocessing` means learned forecasts of the full spectrum at a buoy **do exist**, as
post-processing of a numerical forecast. The layer-2 wording in `../CLAUDE.md` §0.3 ("none
forecast the full spectrum") therefore needs qualifying. What survives is this: no learned model
found forecasts the spectrum from a station's own history, without a numerical forecast as input.
`01_introduction.tex` already ends its spectrum paragraph on that narrower claim, so it holds. The
paragraph's list of prior "lines" of spectral work (estimation; spatial prediction from forcing)
would be more complete with post-processing added as a third. Separately, their hyperparameter
search did not select buoy history as an input. A reviewer may put this to the manuscript; it is
sourced from the snapshot docs only, so check it in the full text before engaging with it.

### Two component claims narrowed on 2026-09-23 — do not overclaim these

The combination above still stands, but two of its components turned out to have precedent that
the 2026-09-18 pass had not found. Both were previously being treated as novel; neither is.
Wording in the manuscript must reflect this.

- **Partition-conditioned evaluation is NOT new.** `hanson2009pacific` (JTECH, 2009) verifies
  three numerical wave models per wind-sea/swell partition and states the motivating premise
  outright — bulk parameters "represent averages over all existing wave systems ... and can mask
  higher-order deficiencies". This is seventeen years earlier than `rogers2025espc`, which had
  been logged as the closest analog. The defensible contribution is now narrower: carrying an
  established *numerical-model verification* practice across into both the evaluation and the
  **training objective** of a learned forecast, where a single aggregate score remains the norm.
  Pair with `ecmwf2019lcwfvparameters` for what standard operational practice actually is (six
  integrated parameters, spectrum not exchanged), with `hernandez2025intercomparison` for its
  present-day scale (WMO Lead Centre, 18 systems), and with `bidlot2002intercomparison` only if
  the historical origin of the practice is wanted.
- **The shape/magnitude factorisation is NOT new in forecasting generally.** `sevlian2018scaling`
  uses exactly "scalar total × normalised shape, multiplied at inference" for electricity load,
  and `guo2026loadshape` argues the same divide-and-conquer rationale this manuscript uses. No
  ocean-domain precedent was found, so the defensible claim is the *application* to wave-spectrum
  forecasting — unit-area spectrum as a decoupled forecast target, with the physical
  m₀ = (Hs/4)² relation supplying magnitude — not the factorisation itself. Note also that
  non-dimensionalising spectra by Hs is routine in wave *characterisation* (JONSWAP-family
  parameterisations), so novelty cannot be claimed for the normalisation either.

**Still clean, uncontested gaps** (unchanged from the 2026-09-18 assessment; no closer prior art
found): (1) Wasserstein/optimal-transport distance as a *training-loss* term for spectral-density
prediction, and (2) DMD as an *auxiliary input feature* rather than as the forecasting method.

### Gap-verification comparators (found via websearch, not locally held)

Full author/venue/volume confirmed; added to `refs.bib` without a `doi` field where none could
be located (never guessed). Full notes at the links below.

- **[meng2023windswell](notes/meng2023windswell.md)** — Meng, Li, Wang, Jiang (2023), *Ocean
  Engineering* 270:113672. Deep learning performs the wind-sea/swell *partitioning itself*;
  closest match for "ML + partition awareness" but does not condition a downstream forecast's
  error metrics on the partition.
- **[breunung2023dmd](notes/breunung2023dmd.md)** — Breunung & Balachandran (2023), *Ocean
  Engineering* 268:113271. Uses DMD as an AR/DMD *forecasting surrogate itself* for
  wave-elevation amplitude series — not as an auxiliary feature layer feeding a separate
  downstream model, which is what this project does. Closest DMD-adjacent precedent found; the
  auxiliary-feature framing itself remains unmatched.
- **[rogers2025espc](notes/rogers2025espc.md)** — Rogers & Janiga (2025), NRL Memorandum Report /
  arXiv:2510.06484. A *numerical* (physics-based) wave-model skill assessment that stratifies
  verification by wind-sea/swell fraction — closest match for "partition-conditioned skill
  scoring," but for a conventional NWP-coupled wave model, not an ML model.

**Two claims came back as clean, uncontested gaps** (no closer prior art found either locally or
via websearch): (1) Wasserstein/optimal-transport distance used as a *training-loss* term
specifically for spectral-density prediction, and (2) DMD used as an *auxiliary input feature*
(vs. DMD as the forecasting method itself). These can be stated as gaps without hedging further;
the other three above should be hedged and cited against their named analogs.

## Pending verification — do NOT cite yet

Found during the gap-verification websearch but with incomplete metadata (missing full author
list, volume, or DOI) — do not add a `refs.bib` entry until confirmed by a follow-up search or a
direct look at the paper itself:

- "A transformer encoder-only framework for multi-horizon wave forecasting with physical and
  window-length interpretability" — *Ocean Engineering*, 2026, PII S0029801826009467. Forecasts
  scalar Hs only (encoder-only, no autoregressive decoder); same-family runner-up to
  `kim2026metoformer`.
- Wang & Jiang, "Physics-guided deep learning for skillful wind-wave modeling," *Science
  Advances*, Dec 2024 — authors/venue/year known, volume/pages/DOI not yet located.
- Deo & Jaiman, "Harnessing Loss Decomposition for Long-Horizon Wave Predictions via Deep Neural
  Networks," NeurIPS ML4PS workshop, 2024 — no DOI/URL located.
- Wedler, Stender, Klein, Ehlers, Hoffmann, "Surface Similarity Parameter," *Neural Networks*,
  2022 — volume/pages/DOI not yet located.
- Serani, Dragone, Stern, Diez et al., ship-motion-in-waves DMD line of work — arXiv:2207.04309
  plus a 2025 *IJACSP* follow-on by Diez et al.; exact follow-on citation not yet confirmed.
- "A frequency-domain and spatiotemporal global joint learning method for wind-swell separation
  and short-term wave field modeling," *Ocean Engineering*, Nov 2025, PII S0029801825030756 — no
  author list located.
- "On the limits of univariate deep learning for significant wave height forecasting" (2026) —
  found only via a citing-works list for `jiang2024comment`; title unconfirmed against a primary
  source.

## Open issues — need a decision from the author

- **RESOLVED 2026-09-29 — `violantecarvalho2009`.** Kept below for the record. Per the author's
  decision, every reference to the γ\* method and threshold now cites both `portilla2009` (the
  source of γ\*) and `violantecarvalho2002` (its precursor). The wording keeps the attribution
  correct, e.g. "proposed by Portilla et al. (2009), building on ... Violante-Carvalho et al.
  (2002)"; a bare grouped citation would imply the 2002 paper states γ\*. Corrected in
  `02_methods.tex`, the parent `CLAUDE.md`, `decisions/log/008`, `utils/spectral_partitioning.py`,
  `utils/spectral_peaks.py`, `nn/evaluate.py`, and `scripts/explore_wave_climate.ipynb`.
  Original entry:
  `sections/02_methods.tex` (and the codebase's own
  `utils/spectral_partitioning.py` comments, per `../CLAUDE.md`) attribute the
  gamma\* = S_obs(fp)/S_PM(fp) > 1 wind-sea/swell threshold to "Violante-Carvalho 2009." A
  dedicated search found two real Violante-Carvalho papers (2002, on wave growth in a
  swell-dominated region; 2004, on swell's influence on wind waves) but **neither states this
  criterion**. The paper that does state it is `portilla2009` (see that note), which reviews and
  recommends the PM peak-ratio method. This looks like a misattribution already present in the
  codebase before this literature pass started, not something introduced here. **No
  `violantecarvalho2009` entry was added to `refs.bib`** — do not add one on a guess. Recommend
  either (a) re-pointing the citation to `portilla2009`, or (b) if there's a specific
  Violante-Carvalho source the author has in mind that this search missed, supplying it directly.
  **Update 2026-09-29:** the author supplied a PDF as the missing source. It is the 2002 paper
  (JOMAE 124(1):14-21), now filed as `violantecarvalho2002`. A full read confirms it does not state
  γ\*: it classifies wind sea by wind direction (±30°) plus a fitted high-frequency level α > 0.001.
  Re-reading `portilla2009` §3b confirms that γ\* > 1.0 is Portilla et al.'s *own* proposed 1D
  algorithm. They developed it after "following the methodology of Violante-Carvalho et al.
  (2002)", which explains the conflated attribution. Option (a) is now the only supported fix:
  cite `portilla2009` for γ\*, optionally with `violantecarvalho2002` as the precursor. The
  "Violante-Carvalho 2009" wording also persists in the parent `CLAUDE.md` and in code comments
  (`utils/spectral_partitioning.py`, `utils/spectral_peaks.py`, `nn/evaluate.py`,
  `scripts/explore_wave_climate.ipynb`).
- **`song2023jpo` page range** is not independently confirmed beyond the article's opening page;
  check the JPO publisher record before quoting a specific page.
- **`kambekar2009asce` page range**: the local PDF itself prints 398-401; Crossref/ASCE Library
  give 398-409. Used 398-409 in `refs.bib` (the DOI-linked record); flag if the PDF's own printed
  range should take precedence.
