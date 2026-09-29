---
key: meng2023windswell
title: Wind-Sea and Swell Separation of 1D Wave Spectrum by Deep Learning
authors: Meng, Y.; Li, X.; Wang, Z.; Jiang, H.
year: 2023
venue: Ocean Engineering, 270, 113672
relevance: medium-high
source_file: none — found via gap-verification websearch, 2026-09-18; no local PDF retrieved
---

# Wind-Sea and Swell Separation of 1D Wave Spectrum by Deep Learning

**Meng, Y., Li, X., Wang, Z., Jiang, H. (2023).** *Ocean Engineering*, 270, 113672.
DOI not located during the websearch pass — omitted from `refs.bib` rather than guessed.

**Provenance note:** this paper was never in `manuscript/literature/` as a PDF. It surfaced
during the Track B gap-verification websearch (2026-09-18) while checking whether any prior work
evaluates an ML wave-spectrum forecast conditioned on wind-sea vs. swell partition classification.
Citation details (author list, venue, volume/pages) come from that search's result snippets, not
from reading the paper directly — treat with slightly less confidence than the 26 papers that
were read PDF-in-hand, and re-verify before quoting anything beyond what's below.

## Summary
Uses deep learning to perform wind-sea/swell partitioning of a 1D wave spectrum itself, validated
against 860k+ buoy/CFOSAT records.

## Relevance to this manuscript
Medium-high — the closest match found for "ML + partition awareness" in the wind-sea/swell
sense this project's evaluation panel uses, but it performs the classification itself as its
primary task; it does not evaluate a separate downstream forecasting model's error conditioned
on that partition, which is what this project's peak-resolved, partition-conditioned diagnostics
do (see `sections/02_methods.tex`, "Peak-resolved, partition-conditioned diagnostics").

## Suggested use
Cite in the gap statement (Introduction) or in the evaluation-methodology paragraph (Methods) as
the closest existing use of learned wind-sea/swell classification, explicitly distinguishing
"classifying the partition" (this paper) from "conditioning a forecast-error metric on the
partition" (this project). Confirm the DOI/volume before final submission — see provenance note
above.
