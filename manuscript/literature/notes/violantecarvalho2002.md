---
key: violantecarvalho2002
title: On the Growth of Wind-Generated Waves in a Swell-Dominated Region in the South Atlantic
authors: Violante-Carvalho, N.; Parente, C.E.; Robinson, I.S.; Nunes, L.M.P.
year: 2002
venue: Journal of Offshore Mechanics and Arctic Engineering, 124(1), 14-21
relevance: medium
source_file: literature/violantecarvalho2002.pdf — published ASME version (8 pp., complete). Supplied by the author 2026-09-29 as `violantecarvalho2009.pdf`; renamed to its real publication year
---

# On the Growth of Wind-Generated Waves in a Swell-Dominated Region in the South Atlantic

**Violante-Carvalho, N., Parente, C.E., Robinson, I.S., Nunes, L.M.P. (2002).** *Journal of
Offshore Mechanics and Arctic Engineering*, 124(1), 14-21. DOI: 10.1115/1.1423636. Metadata
confirmed 2026-09-29 against Crossref (`api.crossref.org/works/10.1115/1.1423636`) and the PDF's
own title page and running footers ("FEBRUARY 2002, Vol. 124"). Crossref's `issued` date is
2001-08-04, the revised-manuscript date. The year used here is the 2002 print issue, which is
also how `portilla2009` cites it. The paper was first presented at OMAE 2001 in Rio de Janeiro.

**This is not a "Violante-Carvalho 2009" paper, and it does not state the γ\* criterion.** The
author supplied it as the missing `violantecarvalho2009` source for the γ\* = S_obs(fp)/S_PM(fp)
> 1 wind-sea/swell threshold in `02_methods.tex`. Reading it in full shows:
- the year is 2002, and it is the same 2002 paper the 2026-09-18 search had already found and
  ruled out (`notes.md`, "Open issues");
- the Pierson-Moskowitz spectrum appears only as background (its Eq. 2). No PM peak-ratio test
  appears anywhere. Its wind-sea classification rests on two different criteria (see Summary).

The γ\* test is `portilla2009`'s own "proposed 1D identification algorithm" (their §3b), stated
there with α_PM = 0.0081, γ = 1, evaluated at f = fp, threshold γ\* > 1.0. Portilla et al. arrive
at it *after* "following the methodology of Violante-Carvalho et al. (2002)". They found that
fitting a JONSWAP spectrum to a wave system's high-frequency part helps identify its peaks, but
"the fitting criterion by itself is not sufficient to decide what is wind sea and what is
swell". That lineage is the likely origin of the mixed-up "Violante-Carvalho 2009" attribution:
Violante-Carvalho's name attached to Portilla's year and criterion.

## Summary
An open-ocean study of wind-sea growth under swell, using 5807 directional spectra from a
heave-pitch-roll buoy moored in over 1000 m of water in Campos Basin, off Rio de Janeiro
(PETROBRAS PROCAP programme, 26 months across 1991-1995). The authors propose a method to fit
and partition multi-peaked 1-D spectra (up to three peaks), as follows:
(i) peaks are selected by three criteria: at least 0.03 Hz apart (twice the frequency
resolution); passing the first 90% confidence-interval test of Guedes Soares and Nolasco (1992);
and a ratio between the two peaks below 15.
(ii) a JONSWAP form with a variable high-frequency exponent n (their Eq. 13) is fitted to the
high-frequency component, subtracted, and the procedure repeated on the remainder.
(iii) a third peak is fitted only if neither of the first two is wind sea.
A peak is classified as wind sea if it lies within ±30° of the buoy-measured wind direction and
its fitted high-frequency level α exceeds 0.001. Both tests are equilibrium-range and wind-based,
not a PM peak ratio.

Findings: the tail exponent n is highly variable, mostly between -4 and -6 (mean -5.2), with no
clear preference for f⁻⁴ or f⁻⁵. For the 243 retained wind-sea cases, α against U10/cp fits
α = 0.0078 (U10/cp)^0.7295 (r = 0.62). This is in close agreement with the JONSWAP relation, so
the authors conclude that swell has no significant effect on wind-sea growth in this open-ocean
region. The site's climatology is itself notable: "about 25 percent of the spectra are unimodal,
whereas the vast majority presents two or more peaks". Spectra with three or more peaks are
"about one-third". The Table 1 breakdown by Hs is an image that was not extracted, so quote only
these two sentences.

## Relevance to this manuscript
Medium. It is not the source of γ\*, so citing it *alone* for γ\* would repeat the
misattribution under a different key. It is cited alongside `portilla2009` as the precursor (see
Suggested use), and is useful for three things:
- **Precursor of the partitioning/identification method.** It is the JONSWAP-fitting wind-sea
  identification that `portilla2009` explicitly builds on before introducing γ\*. It is citable
  as the precursor ("building on the fitting approach of Violante-Carvalho et al. (2002),
  Portilla et al. (2009) proposed..."), never as the source of the threshold.
- **An alternative published peak-significance test** (0.03 Hz separation, peak ratio < 15,
  confidence-interval trough test). Portilla et al. discuss it and criticise it as analogous to
  a contrast criterion with the same limitations. That is useful context if a reviewer asks why
  the Portilla four-criterion test was chosen over other published options.
- **Evidence that multimodal spectra dominate in swell-dominated open-ocean regions.** Only
  about 25% of 5807 buoy spectra were unimodal, and about a third had three or more peaks. This
  backs the claim that bimodality is common, not a corner case, which motivates the
  partition-conditioned evaluation panel. It is a single South Atlantic site, so hedge it as
  one well-documented example, not a global rate.
Its opening line also states the engineering motivation ("Detailed knowledge of the shape of the
ocean wave spectrum and its growth is important information for offshore engineering
purposes", e.g. loads on marine structures and floating-body response). That is a framing claim
in the introduction, not a result.

## Suggested use
**Author's decision (2026-09-29):** whenever the manuscript refers to the γ\* method or
threshold, cite both `portilla2009` and `violantecarvalho2002`. Word it so the attribution stays
correct: Portilla proposed γ\*, building on this paper's fitting approach. `02_methods.tex`
now reads "this peak-ratio criterion was proposed by \citet{portilla2009}, building on the
spectral-fitting wind-sea identification of \citet{violantecarvalho2002}". Avoid a bare
`\citep{portilla2009,violantecarvalho2002}` directly after the γ\* definition, which would imply
this paper states it. Introduction or
Discussion: cite it for the prevalence of multimodal spectra in an open-ocean, swell-dominated
region (about 25% unimodal), alongside `orimoloye2019bimodal`'s argument that composition
matters at fixed energy. Do not cite it for anything about the PM peak ratio.
