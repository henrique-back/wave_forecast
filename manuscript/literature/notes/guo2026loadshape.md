---
key: guo2026loadshape
title: Research on multi-algorithm fusion methods for forecasting typical daily load profiles in power system planning for target years
authors: Guo, Q.; Zhang, Y.; Hao, Q.; Wang, C.; Zhang, Y.; Hu, S.; Hou, K.
year: 2026
venue: Frontiers in Energy Research, 14, article 1699137
relevance: medium
source_file: literature/guo2026loadshape.pdf (Frontiers open access, CC-BY, 13 pp — complete)
---

# Multi-algorithm fusion for forecasting typical daily load profiles

**Guo, Q., Zhang, Y., Hao, Q., Wang, C., Zhang, Y., Hu, S., Hou, K. (2026).** *Frontiers in
Energy Research*, 14, 1699137. DOI: 10.3389/fenrg.2026.1699137.

## Summary
Forecasts typical daily electricity load profiles by splitting the problem in two: ARIMA
forecasts the daily **peak** load, a Random Forest forecasts the **normalised** load curve, and
the two are recombined multiplicatively. The split is argued explicitly as a divide-and-conquer
design, on the grounds that "ARIMA is peak-optimal, while RF is shape-optimal" — i.e. the two
sub-problems have different character and are better served by different models.

## Relevance to this manuscript
Medium. Where `sevlian2018scaling` establishes that the factorisation *exists* as standard
practice, this one is the closer match in **reasoning**: it makes the same argument this
manuscript makes for its shape/magnitude split — that magnitude and shape are different
problems, best decoupled and modelled separately, then recombined. That the same reasoning
arises independently in an unrelated forecasting domain is a point in the design's favour and
worth saying so.

Differences to state if cited: normalisation is by the daily **maximum**, not by area, so there
is no conserved-mass/energy interpretation (this manuscript's unit-area spectrum integrates to
one by construction, and its magnitude carries the physical m₀ = (Hs/4)² meaning); the task is
long-horizon planning for a target year, not short-lead operational forecasting; and there is no
autoregressive or transformer machinery involved.

## Suggested use
Discussion, alongside `sevlian2018scaling`, where the shape/magnitude split is justified — as
independent cross-domain support for the *reasoning* behind decoupling the two sub-problems,
rather than as a methodological precedent to follow.
