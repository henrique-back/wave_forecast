---
key: portilla2009
title: Spectral Partitioning and Identification of Wind Sea and Swell
authors: Portilla, J.; Ocampo-Torres, F.J.; Monbaliu, J.
year: 2009
venue: Journal of Atmospheric and Oceanic Technology, 26(1), 107-122
relevance: high
source_file: literature/portilla2009.pdf
---

# Spectral Partitioning and Identification of Wind Sea and Swell

**Portilla, J., Ocampo-Torres, F.J., Monbaliu, J. (2009).** *Journal of Atmospheric and Oceanic
Technology*, 26(1), 107-122. DOI: 10.1175/2008JTECHO609.1.

## Summary
Reviews and compares 1D and 2D wave-spectrum partitioning techniques (watershed algorithm +
combining/thresholding rules) and wind-sea/swell identification methods (the NDBC steepness
method vs. the Pierson-Moskowitz peak-ratio method), proposing a digital-filter-based 2D
partitioning scheme and finding the PM peak-ratio method more consistent for identification,
tested on NDBC buoy 41013 data.

## Relevance to this manuscript
High — this is the direct methodological source for the codebase's `find_significant_peaks`
(four-criterion significant-peak test) and appears to be the actual origin of the PM peak-ratio
wind-sea/swell classification, currently mis-attributed in `02_methods.tex`/`CLAUDE.md` to a
"Violante-Carvalho 2009" paper that a dedicated search could not confirm exists with this
criterion (see `manuscript/literature/notes.md`, "Open issues").

## Suggested use
Primary citation for the peak-detection/partitioning methodology in Methods/Evaluation. Given
the Violante-Carvalho attribution issue, this paper is the strongest candidate to *also* cover
the gamma* = S_obs(fp)/S_PM(fp) wind-sea/swell threshold, pending a decision from the author on
how to correct the existing `violantecarvalho2009` citation key.

**Update 2026-09-29 (re-read of §3b):** γ\* is this paper's *own* "proposed 1D identification
algorithm", not a method it merely reviews. It is the ratio of a wave system's peak energy to a
PM spectrum at the same fp (JONSWAP Eq. 6 with γ = 1, α = α_PM = 0.0081), and γ\* > 1.0 means wind
sea. It is introduced after "following the methodology of Violante-Carvalho et al. (2002)"
(`violantecarvalho2002`), whose JONSWAP fitting alone "is not sufficient to decide what is wind
sea and what is swell". So this paper is the correct and sole source for the threshold.
`violantecarvalho2002` is the precursor only. Per the author's decision, the manuscript cites
both whenever it refers to the γ\* method (see `violantecarvalho2002.md`, Suggested use).
