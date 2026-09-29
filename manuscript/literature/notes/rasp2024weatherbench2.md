---
key: rasp2024weatherbench2
title: "WeatherBench 2: A Benchmark for the Next Generation of Data-Driven Global Weather Models"
authors: Rasp, S.; Hoyer, S.; Merose, A.; Langmore, I.; Battaglia, P.; Russell, T.; Sanchez-Gonzalez, A.; Yang, V.; Carver, R.; Agrawal, S.; Chantry, M.; Ben Bouallegue, Z.; Dueben, P.; Bromberg, C.; Sisk, J.; Barrington, L.; Bell, A.; Sha, F.
year: 2024
venue: Journal of Advances in Modeling Earth Systems, 16(6), e2023MS004019
relevance: medium-high
source_file: literature/rasp2024weatherbench2.pdf — VERSION MISMATCH, this is the arXiv:2308.15560 v2 preprint (44 pp), not the published JAMES article; cite the JAMES version, verify quotes against it
---

# WeatherBench 2: A Benchmark for the Next Generation of Data-Driven Global Weather Models

**Rasp, S., Hoyer, S., Merose, A., et al. (2024).** *Journal of Advances in Modeling Earth
Systems*, 16(6), e2023MS004019. DOI: 10.1029/2023MS004019. CC-BY.

This is the paper behind the arXiv link in `personal_notes.md`'s "weather - numerical vs AI
models" entry (arXiv:2308.15560 = WeatherBench 2). It has a peer-reviewed JAMES version, so the
journal article is what should be cited, not the preprint.

## Summary
An open evaluation framework and continuously-updated leaderboard for benchmarking data-driven
global weather models against physical (NWP) models on common ground: shared ground truth,
shared metrics, shared verification protocol. The abstract describes the design principles and
"presents results for current state-of-the-art physical and data-driven weather models."

## Relevance to this manuscript
Medium-high, but **with an important wording constraint**. It is the right *kind* of source for
the Introduction's opening claim — formal, comparative, community-standard, not a single
model-announcement paper — and it is the published counterpart of the benchmark/leaderboard the
author had in mind.

**It does not license the phrase "consistently outperform."** The abstract is deliberately
neutral: it presents results for both model families and explicitly "discuss[es] caveats in the
current evaluation setup". Anyone citing it for a strong superiority claim is over-reading it.
See `benbouallegue2024rise` for the same constraint from the operational side, and the
`personal_notes.md` entry "weather - numerical vs AI models" for the recommended softened wording.

It is also **atmosphere-only** — it says nothing whatever about ocean or wave forecasting. The
second half of the author's claim ("gains have been slower to materialise for ocean
forecasting") is not supported by this reference and needs separate support.

## Suggested use
Introduction, first paragraph — the citation for "data-driven models are now formally evaluated
head-to-head against physical NWP models", paired with `benbouallegue2024rise`. Cite for the
existence and maturity of the comparison, not for a verdict on it.
