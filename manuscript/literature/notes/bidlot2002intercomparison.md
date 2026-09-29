---
key: bidlot2002intercomparison
title: Intercomparison of the Performance of Operational Ocean Wave Forecasting Systems with Buoy Data
authors: Bidlot, J.-R.; Holmes, D.J.; Wittmann, P.A.; Lalbeharry, R.; Chen, H.S.
year: 2002
venue: Weather and Forecasting, 17(2), 287-310
relevance: medium-high
source_file: literature/bidlot2002intercomparison.pdf — complete, all 24 pages (287–310, Tables 1–9, Figs. 1–12, references). Re-checked 2026-09-29 with pdfinfo. Earlier versions of this note called the copy partial (pp. 287–296 only); that was wrong.
---

# Intercomparison of the Performance of Operational Ocean Wave Forecasting Systems with Buoy Data

**Bidlot, J.-R., Holmes, D.J., Wittmann, P.A., Lalbeharry, R., Chen, H.S. (2002).** *Weather and
Forecasting*, 17(2), 287-310. DOI: 10.1175/1520-0434(2002)017<0287:IOTPOO>2.0.CO;2.

## Summary
Five operational forecasting centres (ECMWF, Met Office, FNMOC, Meteorological Service of
Canada, NCEP) exchange wave-model output monthly and verify it against moored buoys and
platforms. Verification is on **scalar fields only**: significant wave height, peak period and
10-m wind speed. The paper notes that even peak period is awkward to compare across centres,
because each derives it from the spectrum differently. This is the founding paper of what became
the WMO Lead Centre for Wave Forecast Verification, so it is the defensible citation for the
claim that operational wave-forecast verification *is* bulk-parameter verification.

**Setup:**
- Period: December 1996 – December 1999.
- Buoys: about 40, all in the northern hemisphere, many of them coastal. Observations are
  averaged over 4 h windows.
- Metrics: bias = model − buoy, RMSE, and SI = standard deviation of the model − buoy difference
  divided by the observed mean.
- NCEP assimilated buoy data from February 1998, so its scores are not independent of the
  buoys.

**Day-2 (48 h) Hs forecasts, Table 4** (checked against the PDF 2026-09-29). Buoy mean
2.49–2.55 m.

| Centre | ECMWF | UKMO | FNMOC | AES | NCEP |
|---|---|---|---|---|---|
| RMSE (m) | 0.60 | 0.76 | 0.65 | 0.68 | 0.72 |
| SI | 0.23 | 0.29 | 0.26 | 0.26 | 0.27 |
| Bias (m) | −0.13 | 0.20 | 0.03 | −0.17 | 0.24 |

pdftotext renders the minus sign as a "2", so ECMWF's bias appears as "20.13" in extracted text;
the value is −0.13. The paper also has:
- analysis-time Hs (Table 2) and analysis-time Tp (Table 6);
- sensitivity runs (Tables 7–9);
- bias and SI at every forecast day 0–5, by region and season, in Figs. 8–11 only. There is no
  24 h table.

It reports no spectral, per-frequency or partition metrics.

## Relevance to this manuscript
Medium-high. It supplies the "what standard practice is" half of the Introduction's evaluation
argument, which `hanson2009pacific` then critiques: operational verification is built on Hs and
period, and a forecast is judged good or bad by scalar scores against buoys. Together the two
give the Introduction a properly sourced progression:
1. standard practice is bulk-parameter verification (this paper);
2. the wave-modelling community itself has documented that this masks spectral deficiencies
   (`hanson2009pacific`);
3. machine-learning wave forecasting has largely inherited the bulk-parameter habit without
   inheriting the critique (this manuscript's gap).

It is also one of only two corpus sources with numbers for a genuine operational physical-model
*forecast* at a stated lead time; the other is `filoche2026postprocessing`. It serves as a
historical reference line for Hs error at 48 h. Before setting it beside this manuscript's
numbers, note three things:
- these are 1996–99 systems, verified on northern-hemisphere buoys;
- the SI definition is bias-removed;
- the table gives Hs only.

## Suggested use
**Superseded in the Introduction as of 2026-09-24.** The bulk-parameter claim is now cited from
the Lead Centre's current documentation (`ecmwf2019lcwfvparameters`, `ecmwf2026lcwfvproject`),
with `hernandez2025intercomparison` for scale. Use this paper only if the manuscript wants the
historical origin of the practice. That link is sourced: the Lead Centre's "Verification results"
page states that its "original quality control and averaging procedure was discussed in Bidlot
et al. (2002)", which backs the "founding paper" claim in the Summary (though only for the
procedure). Optionally, cite Table 4 in the Discussion as a historical 48 h Hs reference line for
operational physical models, with the caveats above. Not needed in Methods.
