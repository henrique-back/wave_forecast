import os
import argparse
import ast
import re
import matplotlib.pyplot as plt

# Path to results folder
base_results_folder = os.path.join(os.path.dirname(__file__), '..', 'results')

# Regexes to extract deltat and lead time from folder names
deltat_pattern = re.compile(r'deltat_(\d+)')
lead_pattern = re.compile(r'lead_(\d+)h')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a validation metric (from best_trial.txt) vs lead time."
    )
    parser.add_argument(
        '--metric', default='val_RMSE',
        help="Metric key stored in best_trial.txt, e.g. val_RMSE, val_Shape_RMSE, "
             "val_Shape_SS, val_overall_SS (default: val_RMSE)."
    )
    parser.add_argument(
        '--experiment', default=None,
        help="Only plot this experiment folder under results/ (default: all)."
    )
    return parser.parse_args()


def _lead_dirs_with_best_trial(parent):
    """Yield (lead_hours, best_trial.txt path) for lead_{N}h dirs directly under parent."""
    for lead_folder in os.listdir(parent):
        lead_match = lead_pattern.fullmatch(lead_folder)
        if not lead_match:
            continue
        best_file = os.path.join(parent, lead_folder, 'best_trial.txt')
        if os.path.exists(best_file):
            yield int(lead_match.group(1)), best_file


def _read_metric(best_file, metric):
    with open(best_file, 'r') as f:
        for line in f:
            if ':' in line:
                key, _, val = line.strip().partition(':')
                key = key.strip()
                if key == metric:
                    try:
                        return ast.literal_eval(val.strip())
                    except (ValueError, SyntaxError):
                        return None
    return None


def collect(metric, experiment_filter):
    """Returns list of (deltat, lead_hours, value), deltat=1 for the flat layout."""
    results = []
    for variable in os.listdir(base_results_folder):
        if experiment_filter is not None and variable != experiment_filter:
            continue
        variable_dir = os.path.join(base_results_folder, variable)
        if not os.path.isdir(variable_dir):
            continue
        for target_folder in os.listdir(variable_dir):
            target_dir = os.path.join(variable_dir, target_folder)
            if not os.path.isdir(target_dir):
                continue

            # Flat layout: lead_{N}h directly under target_dir.
            for lead_hours, best_file in _lead_dirs_with_best_trial(target_dir):
                v = _read_metric(best_file, metric)
                if v is not None:
                    results.append((1, lead_hours, v))

            # Nested layout: deltat_{N}/lead_{N}h under target_dir.
            for deltat_folder in os.listdir(target_dir):
                deltat_match = deltat_pattern.fullmatch(deltat_folder)
                if not deltat_match:
                    continue
                deltat = int(deltat_match.group(1))
                deltat_dir = os.path.join(target_dir, deltat_folder)
                for lead_hours, best_file in _lead_dirs_with_best_trial(deltat_dir):
                    v = _read_metric(best_file, metric)
                    if v is not None:
                        results.append((deltat, lead_hours, v))
    return results


def main():
    args = parse_args()
    results = collect(args.metric, args.experiment)

    if not results:
        print(f"No results found for metric={args.metric!r} "
              f"experiment={args.experiment!r}.")
        return

    results.sort(key=lambda x: (x[0], x[1]))

    grouped = {}
    for deltat, lead_hours, value in results:
        grouped.setdefault(deltat, []).append((lead_hours, value))

    plt.figure(figsize=(8, 5))
    for deltat, values in grouped.items():
        values.sort(key=lambda x: x[0])
        hours = [v[0] for v in values]
        metric_values = [v[1] for v in values]
        plt.plot(hours, metric_values, marker='o', label=f"Δt = {deltat}h")

    plt.xlabel("Lead Time (hours)")
    plt.ylabel(args.metric)
    title = f"{args.metric} vs Lead Time"
    if args.experiment:
        title += f" — {args.experiment}"
    plt.title(title)
    plt.grid(True)
    plt.legend(title="Sampling Interval")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
