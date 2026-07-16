"""
Scan results/ for experiment folders and regenerate results/RESEARCH_LOG.md.

Run manually:
    python scripts/summarize_results.py

Also called automatically at the end of optimize.py.
"""
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent.parent / 'results'
LOG_FILE = RESULTS_DIR / 'RESEARCH_LOG.md'

# Metrics shown in the per-experiment table, in display order.
# (key in best_trial.txt, display label)
METRICS = [
    ('val_overall_SS',        'SS ↑'),
    ('val_Hs_RMSE',           'Hs RMSE ↓'),
    ('val_Tm02_RMSE',         'Tm02 RMSE ↓'),
    ('val_Shape_RMSE',        'Shape RMSE ↓'),
    ('val_Shape_SS',          'Shape SS ↑'),
    ('val_Shape_Mass_Error',  'Shape Mass Err ↓'),
    ('val_SI_mean',           'SI mean ↓'),
]

# Primary metric used in the cross-experiment comparison table.
PRIMARY_METRIC = 'val_Shape_RMSE'
PRIMARY_LABEL  = 'Shape RMSE'


def _parse_scalar(text: str, key: str):
    m = re.search(rf'^{re.escape(key)}: ([^\[\n]+)', text, re.MULTILINE)
    if m:
        try:
            return float(m.group(1).strip())
        except ValueError:
            pass
    return None


def parse_best_trial(path: Path) -> dict:
    text = path.read_text()
    result = {}
    for key, _ in METRICS:
        v = _parse_scalar(text, key)
        if v is not None:
            result[key] = v
    # Also capture the objective metric and lead time for context
    for key in ('val_RMSE', 'val_Bias'):
        v = _parse_scalar(text, key)
        if v is not None:
            result[key] = v
    return result


def read_metadata(exp_dir: Path) -> dict:
    meta_path = exp_dir / 'metadata.md'
    if not meta_path.exists():
        return {}
    text = meta_path.read_text()
    result = {}
    for field, key in [
        ('Date',             'date'),
        ('Description',      'description'),
        ('OBJECTIVE_METRIC', 'objective'),
        ('STUDY_VERSION',    'study_version'),
        ('CHANNEL_SET',      'channel_set'),
        ('AUX_SET',          'aux_set'),
        ('Architecture',     'architecture'),
    ]:
        m = re.search(rf'\*\*{field}\*\*[:\s]+(.+)', text)
        if m:
            result[key] = m.group(1).strip()
    return result


def find_experiments():
    """Return list of (name, Path) for all experiment folders, sorted by date in metadata."""
    experiments = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir() and (d / 'metadata.md').exists():
            experiments.append((d.name, d))
    return experiments


def _collect_lead_dirs(parent: Path) -> dict:
    """Returns {lead_time_h: best_trial.txt Path} for lead_{N}h dirs directly under parent."""
    found = {}
    for lead_dir in parent.iterdir():
        if not lead_dir.is_dir():
            continue
        m = re.match(r'lead_(\d+)h', lead_dir.name)
        if not m:
            continue
        trial_file = lead_dir / 'best_trial.txt'
        if trial_file.exists():
            found[int(m.group(1))] = trial_file
    return found


def collect_results(exp_dir: Path) -> dict:
    """Returns {target: {lead_time_h: {metric: value}}} for all targets/leads found.

    Keyed by target (not just lead time) because an experiment folder can
    hold more than one target subdirectory (e.g. hs_shape_v5/{hs,shape}) —
    merging them by lead time alone would silently let one target's metrics
    clobber the other's at the same lead time.

    Supports both the current flat layout (results/{exp}/{target}/lead_{N}h/,
    written by scripts/optimize.py) and the older deltat_{N}-nested layout
    (results/{exp}/{target}/deltat_{N}/lead_{N}h/) — mirrors the same
    fallback nn/checkpoints.py::find_checkpoint uses for loading checkpoints.
    """
    results = {}
    for target_dir in exp_dir.iterdir():
        if not target_dir.is_dir() or target_dir.name == '__pycache__':
            continue

        target_results = {}

        # Flat layout: lead_{N}h directly under target_dir.
        for lead_h, trial_file in _collect_lead_dirs(target_dir).items():
            target_results[lead_h] = parse_best_trial(trial_file)

        # Nested layout: deltat_{N}/lead_{N}h under target_dir.
        for deltat_dir in target_dir.iterdir():
            if not deltat_dir.is_dir() or not re.match(r'deltat_\d+', deltat_dir.name):
                continue
            for lead_h, trial_file in _collect_lead_dirs(deltat_dir).items():
                target_results[lead_h] = parse_best_trial(trial_file)

        if target_results:
            results[target_dir.name] = target_results
    return results


def _flatten_primary_metric(results: dict) -> dict:
    """{target: {lead_h: metrics}} → {lead_h: value} for PRIMARY_METRIC.

    Merges across targets since PRIMARY_METRIC (Shape RMSE) is normally only
    populated by one target (shape, or density) per experiment; if more than
    one target has a value at the same lead, the first one found wins.
    """
    flat = {}
    for target_results in results.values():
        for lead_h, metrics in target_results.items():
            if lead_h not in flat and PRIMARY_METRIC in metrics:
                flat[lead_h] = metrics[PRIMARY_METRIC]
    return flat


def fmt(v, better='lower'):
    if v is None:
        return '—'
    return f'{v:.4f}'


def build_log(experiments: list) -> str:
    lines = []
    lines.append('# Wave Forecast — Research Log')
    lines.append('')
    lines.append(f'*Auto-generated by `scripts/summarize_results.py` — {datetime.now().strftime("%Y-%m-%d %H:%M")}.*')
    lines.append('')

    if not experiments:
        lines.append('No experiments found under `results/`.')
        return '\n'.join(lines)

    # ── Per-experiment section ──────────────────────────────────────────────
    # exp_name → {target → {lead_h → metrics}}
    all_target_results: dict[str, dict] = {}

    for exp_name, exp_dir in experiments:
        meta = read_metadata(exp_dir)
        results = collect_results(exp_dir)
        all_target_results[exp_name] = results

        lines.append(f'---')
        lines.append('')
        lines.append(f'## {exp_name}')
        lines.append('')
        lines.append(f'| Field | Value |')
        lines.append(f'|-------|-------|')
        lines.append(f'| Date | {meta.get("date", "?")} |')
        lines.append(f'| Objective | {meta.get("objective", "?")} |')
        lines.append(f'| Study version | {meta.get("study_version", "?")} |')
        lines.append(f'| Channel set | {meta.get("channel_set", "?")} |')
        lines.append(f'| Aux set | {meta.get("aux_set", "?")} |')
        lines.append(f'| Architecture | {meta.get("architecture", "?")} |')
        lines.append(f'| Description | {meta.get("description", "?")} |')
        lines.append('')

        if not results:
            lines.append('*No `best_trial.txt` files found for this experiment.*')
            lines.append('')
            continue

        # One table per target subdirectory (e.g. hs_shape_v5 has both 'hs'
        # and 'shape') — merging them by lead time alone would let one
        # target's metrics silently clobber the other's.
        for target_name in sorted(results.keys()):
            target_results = results[target_name]
            lines.append(f'**target: `{target_name}`**')
            lines.append('')

            available_leads = sorted(target_results.keys())
            header = '| Metric |' + ''.join(f' {h}h |' for h in available_leads)
            sep    = '|:-------|' + ''.join('------:|' for _ in available_leads)
            lines.append(header)
            lines.append(sep)
            for metric_key, metric_label in METRICS:
                row = f'| {metric_label} |'
                for h in available_leads:
                    v = target_results.get(h, {}).get(metric_key)
                    row += f' {fmt(v)} |'
                lines.append(row)
            lines.append('')

    # ── Cross-experiment comparison (only when > 1 experiment) ─────────────
    if len(experiments) > 1:
        lines.append('---')
        lines.append('')
        lines.append(f'## Cross-Experiment Comparison — {PRIMARY_LABEL}')
        lines.append('')
        lines.append(f'Lower is better. Blank = no result yet.')
        lines.append('')

        # Flatten each experiment's per-target results down to
        # {lead_h: PRIMARY_METRIC value} — only one target normally
        # populates PRIMARY_METRIC (Shape RMSE) per experiment.
        flat_results = {
            exp_name: _flatten_primary_metric(results)
            for exp_name, results in all_target_results.items()
        }

        all_leads = sorted(set(
            h for r in flat_results.values() for h in r.keys()
        ))
        exp_names = [name for name, _ in experiments]

        header = '| Lead |' + ''.join(f' {n} |' for n in exp_names)
        sep    = '|------|' + ''.join('------:|' for _ in exp_names)
        lines.append(header)
        lines.append(sep)

        for h in all_leads:
            row = f'| {h}h |'
            best_v = min(
                (r.get(h) for r in flat_results.values() if r.get(h) is not None),
                default=None
            )
            for name, _ in experiments:
                v = flat_results[name].get(h)
                cell = fmt(v)
                if v is not None and v == best_v:
                    cell = f'**{cell}**'
                row += f' {cell} |'
            lines.append(row)

        lines.append('')

    return '\n'.join(lines)


def main():
    experiments = find_experiments()
    log = build_log(experiments)
    LOG_FILE.write_text(log)
    exp_names = [e[0] for e in experiments]
    print(f'Research log written → {LOG_FILE}')
    print(f'Experiments found: {exp_names}')


if __name__ == '__main__':
    main()
