---
key: hanson2009pacific
title: Pacific Hindcast Performance of Three Numerical Wave Models
authors: Hanson, J.L.; Tracy, B.A.; Tolman, H.L.; Scott, R.D.
year: 2009
venue: Journal of Atmospheric and Oceanic Technology, 26(8), 1614-1633
relevance: high
source_file: literature/hanson2009pacific.pdf (20 pp, complete — matches the 1614-1633 page range)
---

# Pacific Hindcast Performance of Three Numerical Wave Models

**Hanson, J.L., Tracy, B.A., Tolman, H.L., Scott, R.D. (2009).** *Journal of Atmospheric and
Oceanic Technology*, 26(8), 1614-1633. DOI: 10.1175/2009JTECHO650.1.

## Summary
Verifies three numerical wave models (WAM, WAVEWATCH III, WAVAD) against Pacific buoy
observations **per spectral partition** rather than only on bulk parameters: non-directional and
directional spectra are partitioned into wind-sea and swell components, and component height,
period and direction are each verified via temporal-correlation and quantile-quantile analyses,
combined into an integrated skill score.

The abstract's opening sentence states this manuscript's own premise, and is quoted here
verbatim from the local PDF (verified, not recalled): "Although mean or integral properties of
wave spectra are typically used to evaluate numerical wave model performance, one must look
into the spectral details to identify sources of model deficiencies." The introduction is more
pointed still: because bulk parameters "represent averages over all existing wave systems, they
provide only a general measure of model performance and can mask higher-order deficiencies."

## Relevance to this manuscript
High, and in two directions at once — it both **supports and constrains** the manuscript's
argument, which is why it must be cited rather than avoided.

*Supports:* it is the published, citable statement that aggregate/bulk verification hides
spectral error, which is the premise the whole peak-resolved, partition-conditioned evaluation
panel rests on. Until now that premise was argued from first principles in `02_methods.tex`
with no citation behind it.

*Constrains:* partition-conditioned verification is **established practice for numerical wave
models**, and has been since 2009 — seventeen years before `rogers2025espc`, which the corpus
had previously logged as the closest analog. So the manuscript must **not** claim to have
invented partition-conditioned evaluation. The honest contribution is narrower and still real:
importing an established numerical-model verification practice into the evaluation *and the
training objective* of a learned forecast, where the field's default remains a single aggregate
error score.

## Suggested use
Introduction — the citation behind "aggregate metrics hide spectral error", and simultaneously
the source that lets the gap be stated honestly ("established for numerical models; not carried
over into machine-learning wave forecasting, where single-score evaluation remains the norm").
Also Methods/Evaluation, alongside `portilla2009`, as precedent for verifying per partition.
Supersedes `rogers2025espc` as the primary citation for this point — that one is an unpublished
NRL memo/preprint and is better kept as a recent corroborating example.
