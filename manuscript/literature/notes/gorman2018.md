---
key: gorman2018
title: Estimation of Directional Spectra from Wave Buoys for Model Validation
authors: Gorman, R.M.
year: 2018
venue: Procedia IUTAM, 26, 81-91 (IUTAM Symposium Wind Waves, London, 4-8 September 2017)
relevance: medium-high
source_file: web search, no local PDF. Open access (hybrid, CC BY-NC-ND 4.0), but ScienceDirect and ResearchGate both return HTTP 403 to automated download; fetch manually from the DOI
---

# Estimation of Directional Spectra from Wave Buoys for Model Validation

**Gorman, R.M. (2018).** *Procedia IUTAM*, 26, 81-91. DOI: 10.1016/j.piutam.2018.03.008.
Metadata confirmed 2026-09-29 against Crossref (type `journal-article`, sole author Richard M.
Gorman) and OpenAlex/Semantic Scholar. The author is at NIWA, Hamilton, New Zealand. The paper
was presented at the IUTAM Symposium Wind Waves (London, September 2017).

**Provenance:** `\citep{gorman2018}` was already in `01_introduction.tex` (the sentence on buoy
directional moments) when this record was made. Nothing in `literature/`, `decisions/` or git
history mentions it, so the key was most likely written from recall. This record identifies the
paper that matches the key and the claim. It does **not** confirm the full sentence (see below).

**Verification depth: abstract only.** The full text could not be retrieved. The abstract below
is verbatim from OpenAlex and matches Semantic Scholar. Nothing beyond the abstract has been
read, so no method detail or number may be attributed to this paper until someone reads it.

## Summary
Abstract, verbatim: "In this paper, we consider the problem of estimating a directional wave
spectrum from 3-dimensional displacement data recorded by a wave buoy. We look at some of the
limitations of existing methods to extend the "first five" directional moments directly
obtainable from such data. With a view to providing the most detailed possible comparisons with
directional spectra obtained from numerical models, we propose the use of a "diagnostic"
directional spectrum, defined to be the closest possible spectrum to a given model spectrum that
satisfies all measured directional moments. This method allows us to quantify the minimum error
in a modelled directional spectrum consistent with a buoy record. The new method is tested on a
range of artificial test cases, and applied to data obtained from a wave buoy deployment off the
New Zealand coast, in conjunction with outputs from a numerical spectral wave model simulation.
It is shown that the method can provide satisfactory results in a wide range of conditions.
Unlike existing approaches, the proposed method can accommodate sea states with more than two
directional peaks, and can assist in removing spurious spectral energy arising from existing
methods for estimating directional spectra from buoy data."

## Relevance to this manuscript
Medium-high. It supports the Introduction's argument for forecasting E(f) rather than the full
directional spectrum. Checked clause by clause against the sentence in `01_introduction.tex`,
"Operational buoys resolve only its first five directional moments, and reconstructing a full
spectrum from them requires estimator-dependent assumptions that fail in precisely the
multi-modal states described above":
- **"only its first five directional moments": supported.** The abstract says the "first five"
  directional moments are what is "directly obtainable" from buoy displacement data.
- **"requires estimator-dependent assumptions": broadly supported.** The abstract discusses
  "limitations of existing methods to extend" the first five moments, and "spurious spectral
  energy arising from existing methods". The word "assumptions" is the manuscript's own gloss.
- **"fail in precisely the multi-modal states described above": overstated.** The abstract's
  claim is narrower: existing approaches cannot "accommodate sea states with more than two
  directional peaks". That concerns *directional* peaks, and more than two of them. The
  Introduction's "multi-modal states described above" are double- or multi-peaked *frequency*
  spectra (mixed seas), and two-peaked cases are not covered by the abstract's claim. The
  corpus also points the other way for two peaks: `violantecarvalho2002` reports that the
  maximum entropy method "produces reasonable results, particularly in the reconstruction of
  directional double-peaked wave spectra" (citing Lygre and Krogstad 1986). The same paper
  says parametric spreading models "are not consistent with reconstructing the two-dimensional
  spectrum S(f, θ) when windsea and swell co-exist, since they attempt to fit a single peak
  centered between the two wave directions" (citing Young 1994).

## Suggested use
Introduction, for the first five directional moments and the limitations and spurious energy of
existing estimators. Either hedge the failure clause to the abstract's scope, e.g. "and existing
estimators cannot accommodate more than two directional peaks and may introduce spurious
energy", or read the full text first and confirm whether it makes a broader claim. Retrieve the
PDF manually before relying on anything beyond the abstract.
