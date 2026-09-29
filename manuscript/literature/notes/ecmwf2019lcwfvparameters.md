---
key: ecmwf2019lcwfvparameters
title: Parameters — WMO Lead Centre for Wave Forecast Verification (LC-WFV)
authors: ECMWF (corporate; Confluence space owned by Richard Mladek)
year: 2019
venue: ECMWF Confluence documentation, https://confluence.ecmwf.int/spaces/WLW/pages/108102915/Parameters
relevance: medium-high
source_file: web page, no local PDF — read directly from the live page via the Confluence REST API on 2026-09-24 (page v5, created 2018-05-16, last updated 2019-12-10). Web page content can change; re-check before submission.
---

# Parameters — WMO Lead Centre for Wave Forecast Verification (LC-WFV)

**ECMWF (2019).** Documentation page of the World Meteorological Organization (WMO) Lead Centre
for Wave Forecast Verification, hosted on ECMWF's Confluence. The author linked the space's
Project page (`ecmwf2026lcwfvproject`); this Parameters page is its sibling in the same space.
`hernandez2025intercomparison` cites the space's front page as its source.

## Summary
This is the Lead Centre's own specification of what participating centres exchange for common
verification. Quoted verbatim: "**6 parameters were agreed for common verification.**" They are:
- atmospheric forcing: 10 metre U wind component and 10 metre V wind component (m/s);
- wave fields: significant height of combined wind waves and swell (m), peak wave period (s), mean
  zero-crossing wave period (s), and mean wave direction (degree true).

The page adds: "The wave parameters are based on the full 2-D spectrum (to avoid ambiguity in the
names coming form [sic] the latest WMO GRIB2 code tables)". So each parameter is *derived from* the
2-D spectrum, and **the spectrum itself is not one of the exchanged fields**. Output is forecast
fields at 0/6/12/18 UTC where available, at 1-6 h output frequency.

Related pages in the same space, read the same day. They back the notes below but are not cited
separately:
- **Verification results** (v4, last updated 2021-07-26). Published scores are the scatter index
  of significant wave height, wave peak period and wind speed only, for the Northern Hemisphere and
  the Mediterranean. The observations are "moored buoys or weather ships and fixed platforms",
  mostly exchanged via the GTS. The page states: "The original quality control and averaging
  procedure was discussed in Bidlot et al. (2002)". This is the sourced link between
  `bidlot2002intercomparison` and the Lead Centre.
- **Front page.** ECMWF "has been designated as the Lead Centre for Wave Forecast Verification
  (LC-WFV) by the World Meteorological Organisation (WMO) Commission for Basic Systems (CBS-2016)".
- **Parameter availability** (last updated 2026-04-22). A per-centre table for the same six
  parameters, covering 18 centres. That count is consistent with the "18 systems" in
  `hernandez2025intercomparison`.

## Relevance to this manuscript
Medium-high. It is the primary, current source for the claim that operational wave-forecast
verification is conducted on bulk parameters. It is also stronger than
`bidlot2002intercomparison` for that claim, for two reasons: it describes present practice rather
than 2002 practice, and it says explicitly that the parameters are integrals of a spectrum that is
not itself exchanged. That second point sets up the Introduction's next paragraph, which argues
that bulk parameters are summaries of a richer quantity.

## Suggested use
Introduction, first paragraph, cited with `hernandez2025intercomparison` (scale: 18 systems) and
`ecmwf2026lcwfvproject` (verification against buoy observations). Keep the wording within what the
page says: six *integrated* parameters, derived from the spectrum, with the spectrum not among
them. Do not write "the Lead Centre never evaluates spectra". The page specifies the common
exchange only, and a centre's in-house verification may go further.
