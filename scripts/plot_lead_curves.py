"""
Lead-time study, step 2 of 2: figures and tables from the per-sample peak
statistics written by scripts/eval_lead_curves.py. Reads only those files —
no model or peak detector runs here — and works with whichever models are
present (phase 1: persistence, AR, GEFS; phase 2 adds the transformer).

  fig1_lead_comparison   peak fidelity vs lead, each model at the lead it was
                         trained for. (a) 00Z start times with GEFS, (b) every
                         hourly start time, without GEFS. 95% paired circular
                         block-bootstrap intervals; transformer = seed mean.
  fig2_extension_<fam>   one curve per trained lead L, solid up to L and dashed
                         beyond, to MAX_STEP; markers at each model's own lead.
                         Persistence for reference. Hourly start times.
  fig3_timeseries_<L>h   rolling-window peak fidelity at lead TS_LEAD against
                         valid time: all samples, multimodal samples, swell
                         partitions, wind-sea partitions; Hs strip on top.
  table_<set>_<L>h       peak fidelity per regime and frequency band.

Per-label rows and panels (swell / wind-sea partitions) are recall x (1 -
min(rel_err, 1)) for that label: precision has no per-label form (decision
031), so they are not the full peak_fidelity — every caption and table says
so. Frequency-band rows are the full score: a predicted peak has its own fp.

Usage:
    python scripts/plot_lead_curves.py
    python scripts/plot_lead_curves.py --ts-lead 96 --table-leads 6 24 96
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.peak_records import pooled_peak_fidelity, N_TRUE, LABELS

project_root = Path(__file__).resolve().parent.parent

FIG_LEADS = [6, 12, 24, 48, 72, 96]
TS_LEAD = 24
TABLE_LEADS = [6, 24, 96]
ROLL_H = 72              # rolling window for fig 3, centred, in hours of valid time
MIN_WINDOW_PEAKS = 10    # fewer true peaks in a window -> gap, not a point
MIN_WINDOW_PEAKS_GEFS = 5   # GEFS pools ~ROLL_H/24 daily cycles per window, not ROLL_H hours
MIN_CELL_PEAKS = 30      # fewer true peaks in a table cell -> blank
N_BOOT = 1000
BLOCK_DAYS = 3

# Categorical slots 1-3 of the dataviz reference palette (validated: CVD
# deltaE >= 9.2); persistence is a neutral reference, not a series hue.
COLORS = {"transformer": "#2a78d6", "ar": "#eb6834", "gefs": "#1baf7a", "persistence": "#8a8a85"}
NAMES = {"transformer": "Transformer", "ar": "Ridge AR", "gefs": "GEFSv12 (control)",
         "persistence": "Persistence"}
# Ordinal blue ramp for trained leads in fig 2 (validated: monotone, adjacent dL >= 0.06).
LEAD_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0a2a55"]
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e4e0"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.5,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class Study:
    def __init__(self, folder):
        t = np.load(folder / "truth.npz", allow_pickle=True)
        self.issue_times = pd.DatetimeIndex(t["issue_times"])
        self.steps = list(t["steps"])
        self.band_edges = t["band_edges"]
        self.n_true_peaks = t["n_true_peaks"]           # (S, n_steps)
        self.dominant_is_swell = t["dominant_is_swell"]
        self.hs = t["hs"]
        self.gefs_rows = t["gefs_rows"]
        self.S = len(self.issue_times)
        # family -> trained lead (None for persistence/gefs) -> list of (stats, rows, meta)
        self.models = defaultdict(lambda: defaultdict(list))
        for path in sorted((folder / "models").glob("*.npz")):
            z = np.load(path, allow_pickle=True)
            meta = json.loads(str(z["meta"]))
            self.models[meta["model"]][meta.get("trained_lead")].append((z["stats"], z["rows"], meta))

    def step_index(self, h):
        return self.steps.index(h)


def block_counts(S, set_rows, block, n_boot, rng):
    """(n_boot, S) multiplicities: circular block bootstrap over the start
    times in set_rows (in time order), zero for rows outside the set."""
    n = len(set_rows)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)[:, :n]
    counts = np.zeros((n_boot, S))
    for b in range(n_boot):
        np.add.at(counts[b], set_rows[idx[b]], 1.0)
    return counts


def score(entries, step, group, set_w, boot_w, label=None, sample_w=None):
    """Seed-mean score and bootstrap 95% CI for one model family at one lead.

    entries : list of (stats, rows, meta) — seeds of one model
    set_w   : (S,) 0/1 start-time set
    boot_w  : (B, S) multiplicities, or None for the point value only
    sample_w: (S,) extra 0/1 sample filter (a regime) or None
    """
    filt = set_w if sample_w is None else set_w * sample_w
    point, draws, comp = [], [], []
    for stats, rows, _ in entries:
        x = stats[:, step, group].astype(np.float64)
        r = pooled_peak_fidelity(x, weights=filt[rows], label=label)
        point.append(r["peak_fidelity"])
        comp.append(r)
        if boot_w is not None:
            bw = boot_w[:, rows] * (1.0 if sample_w is None else sample_w[rows])
            draws.append(pooled_peak_fidelity(x, weights=bw, label=label)["peak_fidelity"])
    lo = hi = np.nan
    if draws:
        d = np.nanmean(np.stack(draws), axis=0)      # seed mean per draw
        if np.isfinite(d).any():
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
    keys = ("precision", "recall", "rel_err", "n_true", "n_pred")
    return {"pf": float(np.mean(point)), "lo": float(lo), "hi": float(hi),
            "pf_seed_std": float(np.std(point)), "n_seeds": len(entries),
            **{k: float(np.mean([c[k] for c in comp])) for k in keys}}


def dedicated(study, family, lead):
    """Entries of `family` evaluated at `lead`: the model trained for it."""
    if family in ("persistence", "gefs"):
        return study.models[family].get(None, [])
    return study.models[family].get(lead, [])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_lead_comparison(study, out, sets, boots):
    fams_by_panel = [("00Z", ["transformer", "ar", "gefs", "persistence"]),
                     ("hourly", ["transformer", "ar", "persistence"])]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    rows = []
    for ax, (set_name, fams) in zip(axes, fams_by_panel):
        for fam in fams:
            xs, ys, los, his = [], [], [], []
            for lead in FIG_LEADS:
                entries = dedicated(study, fam, lead)
                if not entries or lead not in study.steps:
                    continue
                r = score(entries, study.step_index(lead), 0, sets[set_name], boots[set_name])
                xs.append(lead); ys.append(r["pf"]); los.append(r["lo"]); his.append(r["hi"])
                rows.append({"set": set_name, "model": fam, "lead_h": lead, **r})
            if not xs:
                continue
            c = COLORS[fam]
            ax.fill_between(xs, los, his, color=c, alpha=0.15, linewidth=0)
            ax.plot(xs, ys, color=c, marker="o", markersize=4, label=NAMES[fam],
                    linestyle="--" if fam == "persistence" else "-")
        n = int(sets[set_name].sum())
        ax.set_title(f"({'a' if set_name == '00Z' else 'b'}) {set_name} start times, n = {n}",
                     loc="left", fontsize=9, color=INK)
        ax.set_xlabel("Lead time (h)")
        ax.set_xticks(FIG_LEADS)
    axes[0].set_ylabel("Peak fidelity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(labels))
    _save(fig, out / "fig1_lead_comparison")
    pd.DataFrame(rows).to_csv(out / "fig1_lead_comparison.csv", index=False)


def fig_extension(study, out, sets):
    pers = study.models["persistence"].get(None, [])
    for fam in ("ar", "transformer"):
        leads = sorted(l for l in study.models[fam] if l is not None)
        if not leads:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.6))
        steps = np.array(study.steps)
        rows = []
        if pers:
            y = [score(pers, i, 0, sets["hourly"], None)["pf"] for i in range(len(steps))]
            ax.plot(steps, y, color=COLORS["persistence"], linestyle=":", label="Persistence")
        for color, lead in zip(LEAD_RAMP[-len(leads):], leads):
            entries = study.models[fam][lead]
            y = np.array([score(entries, i, 0, sets["hourly"], None)["pf"] for i in range(len(steps))])
            rows += [{"model": fam, "trained_lead_h": lead, "step_h": int(h), "pf": float(v)}
                     for h, v in zip(steps, y)]
            inside = steps <= lead
            extra = ""
            if fam == "ar":
                extra = f" (order {entries[0][2]['order']})"
            ax.plot(steps[inside], y[inside], color=color, label=f"trained {lead} h{extra}")
            ax.plot(steps[steps >= lead], y[steps >= lead], color=color, linestyle="--")
            if lead in study.steps:
                ax.plot([lead], [y[study.step_index(lead)]], marker="o", markersize=5, color=color,
                        markeredgecolor="white", markeredgewidth=1)
        ax.set_xlabel("Forecast step (h)")
        ax.set_ylabel("Peak fidelity")
        ax.set_xticks(FIG_LEADS)
        ax.set_title(f"{NAMES[fam]}: solid up to the trained lead, dashed beyond "
                     f"(hourly start times, n = {int(sets['hourly'].sum())})", loc="left", fontsize=9, color=INK)
        ax.legend(loc="lower left", fontsize=8)
        _save(fig, out / f"fig2_extension_{fam}")
        pd.DataFrame(rows).to_csv(out / f"fig2_extension_{fam}.csv", index=False)


def _rolling(stats, rows, centers, half):
    """Sum stats over rows within +-half of each center (all in truth-row units)."""
    order = np.argsort(rows)
    rows, stats = rows[order], stats[order]
    csum = np.concatenate([np.zeros((1,) + stats.shape[1:]), np.cumsum(stats, axis=0)])
    lo = np.searchsorted(rows, centers - half, side="left")
    hi = np.searchsorted(rows, centers + half, side="right")
    return csum[hi] - csum[lo]                                     # (n_centers, N_STATS)


def fig_timeseries(study, out, lead):
    if lead not in study.steps:
        print(f"fig3: lead {lead} h not a scored step, skipped")
        return
    i = study.step_index(lead)
    valid = study.issue_times + pd.Timedelta(hours=lead)
    multimodal = (study.n_true_peaks[:, i] >= 2).astype(float)
    panels = [("All samples — peak fidelity", None, None),
              ("Multimodal samples (≥2 true peaks) — peak fidelity", None, multimodal),
              ("Swell partitions — recall × (1 − rel. error)", "swell", None),
              ("Wind-sea partitions — recall × (1 − rel. error)", "wind_sea", None)]
    fams = [f for f in ("transformer", "ar", "gefs", "persistence") if dedicated(study, f, lead)]
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(9, 9), sharex=True,
                             gridspec_kw={"height_ratios": [0.6] + [1] * len(panels)})
    ax0 = axes[0]
    ax0.plot(valid, study.hs[:, i], color=INK, linewidth=1)
    ws = ~study.dominant_is_swell[:, i] & (study.n_true_peaks[:, i] > 0)
    ax0.fill_between(valid, 0, 1, where=ws, transform=ax0.get_xaxis_transform(), color=GRID,
                     linewidth=0, label="highest true peak is wind sea")
    ax0.set_ylabel("Hs (m)")
    ax0.legend(loc="upper right", fontsize=8)
    half = ROLL_H // 2
    rows_out = []
    for ax, (title, label, sample_w) in zip(axes[1:], panels):
        for fam in fams:
            curves = []
            for stats, rows, _ in dedicated(study, fam, lead):
                x = stats[:, i, 0].astype(np.float64)
                if sample_w is not None:
                    x = x * sample_w[rows][:, None]
                centers = rows if fam == "gefs" else np.arange(study.S)
                tot = _rolling(x, rows, centers, half)
                r = pooled_peak_fidelity(tot[:, None, :], label=label)
                min_peaks = MIN_WINDOW_PEAKS_GEFS if fam == "gefs" else MIN_WINDOW_PEAKS
                y = np.where(r["n_true"] >= min_peaks, r["peak_fidelity"], np.nan)
                curves.append((centers, y))
            centers = curves[0][0]
            y = np.nanmean(np.stack([c[1] for c in curves]), axis=0) if len(curves) > 1 else curves[0][1]
            if fam == "gefs":
                ax.plot(valid[centers], y, linestyle="none", marker="o", markersize=3,
                        color=COLORS[fam], label=NAMES[fam])
            else:
                ax.plot(valid[centers], y, color=COLORS[fam], label=NAMES[fam],
                        linewidth=1.0 if fam == "persistence" else 1.3,
                        linestyle=":" if fam == "persistence" else "-")
            rows_out += [{"panel": title, "model": fam, "valid_time": t, "value": v}
                         for t, v in zip(valid[centers], y)]
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_ylim(0, 1)
    axes[-1].set_xlabel(f"Valid time — lead {lead} h, centred {ROLL_H} h windows (gaps: < {MIN_WINDOW_PEAKS} "
                        f"true peaks)\nGEFS dots pool the ~{ROLL_H // 24} daily cycles in each window "
                        f"(gaps: < {MIN_WINDOW_PEAKS_GEFS} true peaks)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(labels))
    _save(fig, out / f"fig3_timeseries_{lead}h")
    pd.DataFrame(rows_out).to_csv(out / f"fig3_timeseries_{lead}h.csv", index=False)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def table(study, out, sets, boots, leads):
    edges = list(study.band_edges)
    bounds = [-np.inf] + edges + [np.inf]
    band_rows = []
    for b in range(len(edges) + 1):
        lo, hi = bounds[b], bounds[b + 1]
        name = (f"fp < {hi:g} Hz" if not np.isfinite(lo) else
                f"fp ≥ {lo:g} Hz" if not np.isfinite(hi) else f"{lo:g} ≤ fp < {hi:g} Hz")
        band_rows.append((name, 1 + b, None, None))
    long_rows = []
    for set_name, fams in (("00Z", ["transformer", "ar", "gefs", "persistence"]),
                           ("hourly", ["transformer", "ar", "persistence"])):
        for lead in leads:
            if lead not in study.steps:
                continue
            i = study.step_index(lead)
            n_true = study.n_true_peaks[:, i]
            specs = [("All samples", 0, None, None),
                     ("Multimodal samples", 0, None, (n_true >= 2).astype(float)),
                     ("Unimodal samples", 0, None, (n_true == 1).astype(float)),
                     ("Swell partitions †", 0, "swell", None),
                     ("Wind-sea partitions †", 0, "wind_sea", None)] + band_rows
            present = [f for f in fams if dedicated(study, f, lead)]
            lines = [f"# Peak fidelity by regime and band — lead {lead} h, {set_name} start times "
                     f"(n = {int(sets[set_name].sum())})", "",
                     "Each model at the lead it was trained for. 95% paired block-bootstrap interval "
                     f"(B = {N_BOOT}, {BLOCK_DAYS}-day blocks); n = true peaks in the cell; blank below "
                     f"{MIN_CELL_PEAKS}.",
                     "† recall × (1 − min(rel. error, 1)) for that label only: precision has no "
                     "per-label form (decision 031).", "",
                     "| Row | " + " | ".join(NAMES[f] for f in present) + " | n |",
                     "|---|" + "---|" * (len(present) + 1)]
            for name, group, label, sample_w in specs:
                cells, n_cell = [], None
                for fam in present:
                    r = score(dedicated(study, fam, lead), i, group, sets[set_name], boots[set_name],
                              label=label, sample_w=sample_w)
                    long_rows.append({"set": set_name, "lead_h": lead, "row": name, "model": fam, **r})
                    n_cell = int(r["n_true"])
                    cells.append("—" if r["n_true"] < MIN_CELL_PEAKS or not np.isfinite(r["pf"])
                                 else f"{r['pf']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
                lines.append(f"| {name} | " + " | ".join(cells) + f" | {n_cell} |")
            (out / f"table_{set_name}_{lead}h.md").write_text("\n".join(lines) + "\n")
    pd.DataFrame(long_rows).to_csv(out / "table_long.csv", index=False)


def _save(fig, stem):
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {stem}.png/.pdf")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", type=Path, default=project_root / "results" / "comparisons" / "lead_curves")
    p.add_argument("--ts-lead", type=int, default=TS_LEAD)
    p.add_argument("--table-leads", type=int, nargs="+", default=TABLE_LEADS)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    return p.parse_args()


def main():
    args = parse_args()
    study = Study(args.folder)
    out = args.folder / "figures"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    hourly = np.arange(study.S)
    sets = {"hourly": np.ones(study.S), "00Z": np.isin(hourly, study.gefs_rows).astype(float)}
    boots = {"hourly": block_counts(study.S, hourly, 24 * BLOCK_DAYS, args.n_boot, rng),
             "00Z": block_counts(study.S, np.sort(study.gefs_rows), BLOCK_DAYS, args.n_boot, rng)}
    fig_lead_comparison(study, out, sets, boots)
    fig_extension(study, out, sets)
    fig_timeseries(study, out, args.ts_lead)
    table(study, out, sets, boots, args.table_leads)


if __name__ == "__main__":
    main()
