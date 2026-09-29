---
key: orimoloye2019bimodal
title: Effects of Swell on Wave Height Distribution of Energy-Conserved Bimodal Seas
authors: Orimoloye, S.; Karunarathna, H.; Reeve, D.E.
year: 2019
venue: Journal of Marine Science and Engineering, 7(3), 79
relevance: high
source_file: literature/orimoloye2019bimodal.pdf (16 pp, complete — supplied by the author 2026-09-23 after MDPI blocked automated download)
---

# Effects of Swell on Wave Height Distribution of Energy-Conserved Bimodal Seas

**Orimoloye, S., Karunarathna, H., Reeve, D.E. (2019).** *Journal of Marine Science and
Engineering*, 7(3), 79. DOI: 10.3390/jmse7030079.

## Summary
A computational (RANS) study of how the wave height distribution of a bimodal sea changes with
the swell percentage and swell peak period, holding the sea state's **total energy content
constant**. An energy-conserved bimodal spectrum is synthesised from unimodal sea states,
converted to a random-wave time series by inverse FFT, and propagated through a RANS model;
wave heights are extracted by down-crossing analysis both near the wavemaker and in shallow
water near a structure. The abstract states the premise directly: "In bimodal seas, swell can
be present at different percentages and different frequencies while the energy content of the
sea state remains unaltered." It concludes that the kurtosis and skewness of the wave height
distribution vary inversely with swell percentage and peak period, and that "non-linearities
are greater in the unimodal seas compared to the bimodal seas with the same energy content."
It motivates the whole exercise on engineering grounds: "An understanding of the wave height
distribution of a sea state is important in forecasting extreme wave height and lifetime
fatigue predictions of marine structures."

**Verification status (updated 2026-09-23):** local PDF now in hand; the abstract sentences
quoted above are confirmed verbatim against it, and the authors are confirmed as Swansea
University (Zienkiewicz Centre / Energy & Environment Research Group). The body beyond the
abstract and introduction has still not been read in detail — do not attribute specific
quantitative results without checking them in the PDF first.

**Scope caveat — this is a coastal, not an offshore-structure, study.** Its framing is coastal
risk (storm waves, sea defences, beaches, shallow-water transformation; it cites a claim that a
bimodal sea state "could be the worst case sea conditions that sea defences or beaches could
experience"), and its outcome variable is the statistical distribution of wave heights, not
vessel motions or mooring loads. Do **not** cite it for a vessel-response or RAO argument. What
it does support is the underlying premise — equal energy content, different spectral composition,
different resulting wave statistics — plus its own stated engineering motivation, quoted above.

## Relevance to this manuscript
High — this is the cleanest available support for the Introduction's core operational premise:
that significant wave height does **not** determine the sea state's engineering consequences,
because two sea states can carry identical total energy (hence identical Hs) while differing in
how that energy is distributed between swell and wind-sea, and behave differently as a result.
That is precisely the argument for forecasting the spectrum rather than a bulk parameter. It
also supplies the "why bimodal specifically" motivation behind this manuscript's peak-resolved,
partition-conditioned evaluation panel: bimodality is not a curiosity, it is the regime in
which bulk parameters are least informative.

Note the direction of its finding is a useful nuance rather than a simple "bimodal is worse"
claim: it reports *greater* non-linearity in unimodal seas at matched energy. The transferable
point for this manuscript is therefore that spectral composition changes behaviour at fixed
energy — not that bimodal seas are uniformly more severe. Do not overstate it in the other
direction.

## Suggested use
Introduction, in the paragraph establishing why the full spectrum is the right forecast target
rather than Hs alone — paired with `simao2025bimodal` (which carries the same premise through
to moored-structure extreme response). Also usable in the Discussion when motivating the
partition-conditioned metrics. Cite for the *premise* (equal energy, different composition,
different behaviour), not for a specific numeric result.
