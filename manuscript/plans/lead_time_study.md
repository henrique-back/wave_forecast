# Lead-time study — plan and runbook

Status 2026-10-02: phase 1 code written and validated on wavetank (see "Validation done");
nothing of phase 1 or 2 has been run for real yet. Phase 2 waits on the loss ablation
(Slurm jobs 93 → 94 → 95 on wavetank) and decision 033.

## What it produces

All on buoy 32012's test split (2017-09-13 → 12-31), shape target, scored with
`peak_fidelity` (decision 031). Output: `results/comparisons/lead_curves/figures/`.

| Output | Content |
|---|---|
| `fig1_lead_comparison` | Peak fidelity at 6/12/24/48/72/96 h: transformer, ridge AR, GEFSv12, persistence. Each model at the lead it was trained for. (a) 00Z start times with GEFS (~100), (b) all hourly start times (~2490), no GEFS. 95% paired block-bootstrap bands; the transformer line is the mean over 5 seeds. |
| `fig2_extension_{ar,transformer}` | One curve per trained lead L, rolled out to 96 h: solid up to L, dashed beyond, marker at L. Persistence for reference. |
| `fig3_timeseries_24h` | Rolling (72 h) peak fidelity against valid time at 24 h lead. Panels: all samples, multimodal samples, swell partitions, wind-sea partitions. Hs strip on top. GEFS as daily dots. |
| `table_{00Z,hourly}_{6,24,96}h.md`, `table_long.csv` | Rows: all / multimodal / unimodal samples, swell / wind-sea partitions, 4 frequency bands. 95% CI in every cell; blank below 30 true peaks. |

Every figure also writes its numbers as CSV next to it.

## Fixed method choices (set before any score was looked at)

- **One start-time set for every model and both phases:** a test row T qualifies if it has
  96 h of test-split history (the largest `seq_len`/AR order either model can pick) and
  T + 96 h is still in the test split. The transformer, added in phase 2, therefore can't
  change the samples the AR and GEFS were scored on.
- **Band and units:** as in `scripts/compare_physical_baseline.py`. Bins 0.0375–0.485 Hz
  (inside the GEFS grid), clipped at 0, renormalised, scaled by the true band m0 so the
  wind-sea/swell labels are physical (decision 029). Truth is the raw density.
- **Scored steps:** every 3 h (GEFS's interval).
- **Per-sample statistics, not per-sample scores** (`utils/peak_records.py`). Peak fidelity is
  a pooled score, and with 1–3 peaks per spectrum a single sample's score is almost always 0
  or 1. The counts behind it add up across samples, so windows, regimes and bootstrap draws
  are re-poolings of those counts. Tests pin exact parity with
  `peak_modality_metrics` + `_compute_val_score`.
- **Regimes:** classifying whole samples doesn't work on this buoy. 90% of test hours are
  multimodal, 185 h are unimodal swell, and only 68 h are unimodal wind sea. So:
  - Swell and wind sea are split **per partition** (3877 swell / 2769 wind-sea partitions).
    Precision has no per-label form (decision 031), so those rows and panels show
    recall × (1 − min(rel. error, 1)) for that label, not the full score. Every caption
    says this. (The alternative, labelling predicted peaks by their own γ\*, would reverse
    part of 031 and is not used.)
  - Multimodal is a **sample** split (≥ 2 true peaks at the valid time) and gets the full score.
    Since it is 90% of samples, it tracks "all samples" closely.
  - **Frequency bands** (fp < 0.08, 0.08–0.125, 0.125–0.2, ≥ 0.2 Hz; edges near the
    true-peak fp quartiles) get the full score, precision included: a predicted peak has
    its own fp.
- **Uncertainty:** circular block bootstrap, 3-day blocks (72 hourly samples or 3 cycles),
  paired across models (the same draws for every model), B = 1000. The transformer interval
  is computed on the seed mean.
- **AR selection:** the AR fits one one-step model and rolls it forward, so "the AR trained
  for lead L" differs only in its order (grid 12/24/48/96). It is now selected the way the
  transformer is: final-step `peak_fidelity` on val (`--objective peak_fidelity`), written
  to `results/linear_baseline_pf/`. The original `results/linear_baseline/`
  (`weighted_mean_SS`-selected, read by `compare_physical_baseline.py`) is untouched. Leads
  that pick the same order give identical forecasts, so their fig-2 curves coincide; the
  legend shows the order.

## Phase 1 — AR, persistence, GEFS (CPU only, no Slurm; internet for GEFS)

```bash
bash slurm/lead_study/phase1_ar.sh        # N_JOBS=8 by default
```

Steps: new tests → AR order selection + test fit at 6–96 h (~20 s per lead) → GEFS fetch to
+96 h (~110 cycles, roughly twice the data per cycle of the 48 h fetch; resumable) →
forecasts + statistics (a few CPU minutes) → figures. Every step resumes when re-run.

Note: the fetch rewrites `buoy_data/32012/gefsv12_c00_spec1d.npz` with leads to 96 h. That is
backwards compatible: `compare_physical_baseline.py` looks leads up by value.

### Checks before phase 2

1. `pytest tests/test_peak_records.py` passes (the script runs it first).
2. At 48 h, fig 1(a) matches `results/comparisons/physical_baseline_gefsv12/lead_48h` on
   their shared cycles: GEFS 0.384, persistence 0.233. (The AR differs only if
   `linear_baseline_pf` picked another order than 12.)
3. Look at the four outputs and settle layout, panels and bands. After this, phase 2 only
   adds rows. Open choices to settle here: time-series lead (24 h; `--ts-lead`), rolling
   window (`ROLL_H = 72`), table leads (`--table-leads`), and whether GEFS belongs in fig 3.
   GEFS's dots pool only ~3 cycles per window, so they are noisy.
4. Wind-sea cells: most have enough peaks at 00Z (106 at 48 h), but watch the blanks.

## Phase 2 — transformer `shape_v14` (GPU)

**Gate:** the loss ablation is settled and decision 033 is written (`scripts/optimize.py`'s v14
search space — `kl_contrib`/`w2_rel`/`peak_rel` and searched head_dim/nhead — is what it
references). If the ablation changes the v14 space, change `optimize.py` before submitting.

With Slurm:
```bash
bash slurm/lead_study/submit_phase2.sh     # LONG_LEAD_TRIALS=40 by default
```
Without Slurm (one free GPU, runs for days):
```bash
nohup bash slurm/lead_study/phase2_local.sh > logs/phase2.log 2>&1 &
```

Job graph: `optimize` at 6/12/24/48 h (80 trials each, independent) → `optimize` at 72/96 h
after 48 h, warm-started with 48 h's 3 best trials (`--warm-start-from 48`) and a 40-trial
budget → `train` per lead (5 seeds, `scripts/train.py` SEEDS) → `eval_transformer`.
`eval_transformer` rolls every seed checkpoint out to 96 h and adds it to the phase-1 statistics,
then redraws everything. Nothing from phase 1 is recomputed.

Things to watch:
- `optimize` jobs hit the 48 h `--time` limit (every v13 lead did). Resubmitting resumes the
  study within its budget, but the `afterok` dependents are lost: submit them by hand.
- 72/96 h trials are the slowest: a 96-step autoregressive decoder, with batch size capped
  at 32/64 above 24 h. Check the first few trials' epoch time before trusting the 48 h limit.
- `train.slurm` gives 24 h for 5 seeds × up to 100 epochs; at 96 h that may be short. If it
  times out, cut `SEEDS` or split it.
- The ablation tuned the loss at 12 h only. If 72/96 h trials land at the edge of the
  `kl_contrib`/`w2_rel`/`peak_rel` ranges, the weights don't carry over to long leads.
  Flag it rather than widening silently.
- `eval_lead_curves.py` asserts that each checkpoint's normalisation matches the loader, that
  inputs don't depend on the target length, and that targets line up with the buoy at T + h.
  If one of these fails, something is wrong. Don't bypass it.

## Moving to another machine

Data, checkpoints and Optuna DBs are tracked in git, so a clone/pull is enough:
1. Commit and push from wavetank (the working tree has uncommitted v14/ablation work: decide
   what goes in), `git pull` on the target.
2. `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
   (install the right torch build first, see requirements.txt).
3. Phase 1 needs internet for GEFS. `downloads/` (the GEFS cache) is git-ignored, so the first
   fetch there starts from scratch.
4. Phase 2 Slurm files use `$SLURM_SUBMIT_DIR`, so no paths to edit. Check `--partition`,
   `--qos` and `--gres` against the target cluster (`sinfo`).
5. Bring results back by committing `results/linear_baseline_pf/`, `results/shape_v14/`,
   `results/comparisons/lead_curves/`, `optuna_study_v14.db` and the updated GEFS npz.

## Validation done (wavetank, 2026-10-02)

- `tests/test_peak_records.py`: 8 tests. Pooled statistics reproduce `peak_modality_metrics`
  and `_compute_val_score` exactly, for the whole batch, a subset, a bootstrap draw, and bands
  summing to the total.
- Pipeline at a 48 h cap (current GEFS file and shape_v13's 48 h checkpoint), against
  `compare_physical_baseline.py`'s saved 48 h arrays on the 103 shared cycles. Peak fidelity
  matched to 6 decimals for all four models: persistence 0.233413, ridge AR 0.228941,
  GEFS 0.384060, shape_v13 0.162723. This covers forecasting, alignment, band, labels and
  the metric.
- AR selection with `--objective peak_fidelity` at 96 h ran in ~20 s (picked order 12).
- Figures and tables rendered and inspected with the AR at 6–48 h.

Side observation from that run (shape_v13, not v14): at 48 h on the 00Z cycles GEFS scores
0.384, persistence 0.233, AR 0.229 and shape_v13 0.163. shape_v13 scores 0 in the two
higher-frequency bands (fp ≥ 0.125 Hz): it predicts no peaks there.
