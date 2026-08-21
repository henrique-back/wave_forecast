"""
Compare the wind-sea/swell peak-fidelity panel across scripts/ablate_loss.py's
5 phases (baseline, kl, wasserstein, peak, combined).

Reads each phase's best_trial.txt (written by save_progress/ablate_loss.py's
own result-writing block) — i.e. exactly the VALIDATION-set numbers that
drove that phase's own weight-selection (see nn/optimization.py::
_compute_val_score's 'peak_fidelity_SS' docstring), not a fresh evaluate()
pass. Fast (no GPU, no data loading), and works incrementally: a phase that
hasn't finished yet (or was pruned/failed before writing best_trial.txt)
just prints as "(pending)" rather than erroring — safe to run while the
last phase is still training.

Metrics compared, each split by partition label (utils.spectral_partitioning
::classify_partition's wind-sea/swell distinction):
    Peak_Height_RelError_{windsea,swell}     lower is better
    Peak_Separation_Recall_{windsea,swell}   higher is better
    Tm02_RMSE_{windsea,swell}                lower is better
    Tm02_Bias_{windsea,swell}                closer to 0 is better

Tm02_Bias is NOT currently recorded: scripts/ablate_loss.py's trial.
set_user_attr list only saves Tm02_RMSE_windsea/_swell (see its
make_objective), not the Bias variants utils.spectral_peaks.
peak_modality_metrics already computes internally — printed as "n/a" here
rather than silently omitted, with a note. Two ways to get it for real:
add 'Tm02_Bias_windsea'/'Tm02_Bias_swell' to that user_attr list (only
affects FUTURE runs — already-completed phases won't have it retroactively,
since only the winning trial's checkpoint was ever saved), or recompute
it directly from each phase's best_model.pt via a fresh evaluate(...,
compute_peak_metrics=True) pass — not done here to keep this script fast
and GPU-free while other phases may still be training.

Also reports, for each of the four metrics above, the wind-sea/swell
AVERAGE (plain arithmetic mean, nan-skipping one label if it's unavailable
— same convention as nn/optimization.py's 'peak_fidelity_SS').

Usage:
    python scripts/compare_ablation_phases.py
    python scripts/compare_ablation_phases.py --lead 12 --study-version lossablation_v2
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.ablate_loss import STUDY_VERSION as DEFAULT_STUDY_VERSION
from scripts.ablate_loss import LEAD_TIME_HOURS as DEFAULT_LEAD_TIME_HOURS
from scripts.ablate_loss import TARGET, PHASE_N_TRIALS

PHASES = list(PHASE_N_TRIALS)  # insertion order: baseline, kl, wasserstein, peak, combined

# (metric prefix, direction) -- direction is purely cosmetic (an arrow next
# to the header), doesn't affect any computation.
METRICS = [
    ("Peak_Height_RelError", "lower better"),
    ("Peak_Separation_Recall", "higher better"),
    ("Tm02_RMSE", "lower better"),
    ("Tm02_Bias", "closer to 0 better"),
]


def _parse_value(raw):
    """best_trial.txt's 'key: value' lines are Python repr()s of whatever
    was passed to trial.set_user_attr — None, an int, or a float (including
    'nan'/'inf'/'-inf', which ast.literal_eval does NOT accept as literals,
    unlike the plain float() builtin used here)."""
    raw = raw.strip()
    if raw == "None":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _phase_dir(phase, study_version, lead_hours):
    return (Path(__file__).parent.parent / "results" / f"lossablation_{phase}_{study_version}"
            / TARGET / f"lead_{lead_hours}h")


def load_phase(phase, study_version, lead_hours):
    """Returns (params: dict, metrics: dict, source: 'test'|'val') for a
    finished phase, or (None, None, None) if it hasn't produced either file
    yet (still running, pruned before completion, or never submitted).

    Prefers test_metrics.json (scripts/evaluate_ablation_phases.py — the
    full evaluate() panel, including Tm02_Bias_windsea/_swell, on the
    held-out TEST set) when present; falls back to best_trial.txt's
    VALIDATION-set numbers (written by ablate_loss.py itself, missing
    Tm02_Bias — see this module's docstring) otherwise. Different phases
    can legitimately be on different sources at once (e.g. right after
    scripts/evaluate_ablation_phases.py has processed some but not all
    phases) — build_report/build_markdown mark each phase's source
    explicitly rather than silently mixing val and test numbers in one
    unlabeled table.
    """
    phase_dir = _phase_dir(phase, study_version, lead_hours)

    test_path = phase_dir / "test_metrics.json"
    if test_path.exists():
        import json
        metrics = json.loads(test_path.read_text())
        params = metrics.get("_trial_params", {})
        return params, metrics, "test"

    val_path = phase_dir / "best_trial.txt"
    if not val_path.exists():
        return None, None, None
    text = val_path.read_text()
    lines = text.splitlines()
    if not any(l.startswith("Best trial parameters:") for l in lines):
        return None, None, None  # file exists but wasn't finished when written

    params = {}
    metrics = {}
    for i, line in enumerate(lines):
        if line.startswith("Best trial parameters:"):
            # Params are a dict literal on the FOLLOWING line.
            import ast
            params = ast.literal_eval(lines[i + 1].strip())
        elif line.startswith("val_"):
            key, _, raw = line.partition(":")
            metrics[key[len("val_"):]] = _parse_value(raw)
    return params, metrics, "val"


def _fmt(x, width=10):
    if x is None:
        return "n/a".rjust(width)
    if isinstance(x, float) and (x != x):  # NaN
        return "nan".rjust(width)
    return f"{x:.4f}".rjust(width)


def _avg(a, b):
    vals = [v for v in (a, b) if v is not None and not (isinstance(v, float) and v != v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _weight_summary(phase, params):
    if not params:
        return "-" if phase == "baseline" else "(pending)"
    return ", ".join(f"{k}={v:.4g}" for k, v in params.items())


def _phase_label(phase, source):
    """Appends [test]/[val] so a table never presents two different splits
    under one unlabeled column — see load_phase's docstring for why a phase
    can legitimately be on either."""
    if source is None:
        return phase
    return f"{phase} [{source}]"


def build_report(study_version, lead_hours):
    rows = {phase: load_phase(phase, study_version, lead_hours) for phase in PHASES}

    lines = []
    lines.append(f"Loss-ablation comparison — target={TARGET}, lead={lead_hours}h, "
                 f"study_version={study_version}")
    lines.append("=" * 78)
    missing = [p for p, (params, _, _) in rows.items() if params is None]
    if missing:
        lines.append(f"(pending / not yet finished: {', '.join(missing)})")
    mixed_sources = len({src for _, _, src in rows.values() if src is not None}) > 1
    if mixed_sources:
        lines.append("([test] = held-out test set, scripts/evaluate_ablation_phases.py; "
                     "[val] = validation set, scripts/ablate_loss.py's own best_trial.txt — "
                     "run scripts/evaluate_ablation_phases.py once every phase is done for "
                     "all-[test], directly comparable numbers)")
    lines.append("")

    lines.append("Winning weight per phase:")
    for phase in PHASES:
        params, _, source = rows[phase]
        lines.append(f"  {_phase_label(phase, source):<18} {_weight_summary(phase, params)}")
    lines.append("")

    for prefix, direction in METRICS:
        lines.append(f"{prefix}  ({direction})")
        header = f"  {'phase':<18} {'windsea':>10} {'swell':>10} {'avg':>10}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for phase in PHASES:
            _, metrics, source = rows[phase]
            label = _phase_label(phase, source)
            if metrics is None:
                lines.append(f"  {label:<18} {'(pending)':>10} {'(pending)':>10} {'(pending)':>10}")
                continue
            windsea = metrics.get(f"{prefix}_windsea")
            swell = metrics.get(f"{prefix}_swell")
            avg = _avg(windsea, swell)
            lines.append(f"  {label:<18} {_fmt(windsea)} {_fmt(swell)} {_fmt(avg)}")
        lines.append("")

    if any(metrics is not None and f"{METRICS[3][0]}_windsea" not in metrics
           for _, metrics, _ in rows.values()):
        lines.append("Note: Tm02_Bias is 'n/a' for any [val]-sourced phase above — "
                     "scripts/ablate_loss.py's saved trial.user_attrs only recorded "
                     "Tm02_RMSE. Run scripts/evaluate_ablation_phases.py to backfill it "
                     "(and switch that phase's whole row to [test]) — see this script's "
                     "module docstring.")

    return "\n".join(lines)


def build_markdown(study_version, lead_hours):
    rows = {phase: load_phase(phase, study_version, lead_hours) for phase in PHASES}
    lines = [f"# Loss-ablation comparison — target={TARGET}, lead={lead_hours}h, "
             f"study_version={study_version}", ""]

    lines.append("## Winning weight per phase")
    lines.append("")
    lines.append("| phase | winning weight |")
    lines.append("|---|---|")
    for phase in PHASES:
        params, _, source = rows[phase]
        lines.append(f"| {_phase_label(phase, source)} | {_weight_summary(phase, params)} |")
    lines.append("")

    for prefix, direction in METRICS:
        lines.append(f"## {prefix} ({direction})")
        lines.append("")
        lines.append("| phase | windsea | swell | avg |")
        lines.append("|---|---|---|---|")
        for phase in PHASES:
            _, metrics, source = rows[phase]
            label = _phase_label(phase, source)
            if metrics is None:
                lines.append(f"| {label} | pending | pending | pending |")
                continue
            windsea = metrics.get(f"{prefix}_windsea")
            swell = metrics.get(f"{prefix}_swell")
            avg = _avg(windsea, swell)
            fmt = lambda v: "n/a" if v is None else (f"{v:.4f}" if not (isinstance(v, float) and v != v) else "nan")
            lines.append(f"| {label} | {fmt(windsea)} | {fmt(swell)} | {fmt(avg)} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lead", type=int, default=DEFAULT_LEAD_TIME_HOURS,
                         help=f"Lead time in hours (default: {DEFAULT_LEAD_TIME_HOURS}, "
                              f"scripts/ablate_loss.py's LEAD_TIME_HOURS)")
    parser.add_argument("--study-version", default=DEFAULT_STUDY_VERSION,
                         help=f"Ablation STUDY_VERSION to read (default: {DEFAULT_STUDY_VERSION!r}, "
                              f"scripts/ablate_loss.py's current value)")
    parser.add_argument("--out", type=str, default=None,
                         help="Also write a Markdown version of this report to this path")
    args = parser.parse_args()

    print(build_report(args.study_version, args.lead))

    if args.out:
        Path(args.out).write_text(build_markdown(args.study_version, args.lead))
        print(f"\nMarkdown report written to {args.out}")


if __name__ == "__main__":
    main()
