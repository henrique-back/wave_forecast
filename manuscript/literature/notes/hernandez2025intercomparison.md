---
key: hernandez2025intercomparison
title: A description of model intercomparison processes and techniques for ocean forecasting
authors: Hernandez, F.; Garcia Sotillo, M.; Melet, A.
year: 2025
venue: State of the Planet, 5-opsr, 17 (Chapter 6.3 of the report "Ocean prediction: present status and state of the art")
relevance: medium
source_file: literature/hernandez2025intercomparison.pdf — full published version (14 pp., Copernicus, CC BY 4.0); metadata confirmed via Crossref DOI lookup 2026-09-24
---

# A description of model intercomparison processes and techniques for ocean forecasting

**Hernandez, F., Garcia Sotillo, M., Melet, A. (2025).** *State of the Planet*, 5-opsr, 17.
DOI: 10.5194/sp-5-opsr-17-2025. Published 2 June 2025. Crossref lists the second author as
given name "Marcos Garcia", family name "Sotillo", and `refs.bib` follows that split.

## Summary
A review chapter on how operational and research ocean models are intercompared. Nearly all of it
is about **ocean circulation**, not waves. It traces the lineage from the atmospheric AMIP/CMIP
projects through the ocean-model efforts (CME, DYNAMO, CORE/OMIP, CLIVAR-GSOP) to the operational
GODAE/OceanPredict framework. It describes the "Class 1-4" metrics: Class 4 compares observations
with the equivalent model forecast at the same time and place, for different lead times. It also
covers the Ocean Reanalysis Intercomparison Project (ORA-IP), ensemble-mean and
consensus-clustering assessment, regional and nested-system intercomparison, and ocean state
monitoring. A worked example (ATL3 SST index, Fig. 2) shows that the ensemble mean beats
individual reanalyses.

**Wave forecasting gets exactly one paragraph** (p. 3, §1). There, routine intercomparison of
wave forecasts "has been settled for many years under the World Meteorological Organization (WMO)
framework". The European Centre for Medium-Range Weather Forecasts (ECMWF) hosts the ongoing
**WMO Lead Centre for Wave Forecast Verification, "where 18 regional and global wave forecast
systems are compared"**, and it keeps an archive of verification statistics so that performance
trends can be tracked over time. The source for this paragraph is the Lead Centre's own web page
(confluence.ecmwf.int/display/WLW, accessed 29 January 2025), so this chapter is a **secondary
source** for the wave claim. One further wave mention is on p. 5: an NCEP evaluation (Campos et
al., 2018) found ensemble wave skill at day 10 outperforming deterministic skill at day 7.

**What it does NOT say:** it never states which variables the Lead Centre verifies. Nothing here
says whether verification is on bulk parameters or on spectra. It does not mention
`bidlot2002intercomparison` or any history of the wave exchange before the Lead Centre.

## Relevance to this manuscript
Medium, for one narrow use. It gives a recent (2025), peer-reviewed citation for the **current
scale and permanence** of operational wave-forecast verification: a WMO-endorsed, ECMWF-hosted,
ongoing multi-system intercomparison of 18 systems. That supports the Introduction's statement
that the bulk-parameter convention is embedded in operational practice.

It **cannot carry the bulk-parameters claim on its own**, because it never names the variables.
The Introduction's argument (§0.3 layer 4 in `../../CLAUDE.md`) needs that claim. **Update
2026-09-24:** the claim is now sourced from the Lead Centre's own documentation, on the pages this
chapter points to. `ecmwf2019lcwfvparameters` lists the six integrated parameters agreed for common
verification, and `ecmwf2026lcwfvproject` covers verification against buoy data. The Introduction
cites Hernandez for scale and those pages for content. It no longer cites
`bidlot2002intercomparison`.

A side note, not a citable finding: p. 4 mentions **"double penalty"** scores, in the context of
comparing re-gridded products that resolve different spatial scales. That is the same verification
pathology that makes per-bin spectral error reward over-smoothing: a peak forecast at a slightly
wrong frequency is penalised both where it is and where it should have been. This chapter is **not**
a source for that argument, because it raises double penalty only in passing and only for spatial
re-gridding. If the manuscript wants to name the concept, it needs a proper forecast-verification
source, and none is in the corpus yet.

## Suggested use
Introduction, first paragraph, as the citation for the present-day scale of operational
wave-forecast verification (WMO Lead Centre, 18 systems), paired with `bidlot2002intercomparison`
for the variables verified. If a Methods sentence ever wants a citation for evaluating a forecast
against point observations across lead times, Class 4 metrics are the relevant concept here, but
that use is optional and low priority. Do not cite it for anything about spectral verification,
wave partitioning, or data-driven models, because it covers none of these.
