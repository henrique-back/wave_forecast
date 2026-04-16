import os
import ast
import re
import matplotlib.pyplot as plt

# ── Configurable ──────────────────────────────────────────────────────────────
# Choose any scalar validation metric stored in best_trial.txt:
#   val_RMSE | val_MAPE | val_CC | val_Bias | val_R2 | val_overall_SS
METRIC = 'val_RMSE'
# ─────────────────────────────────────────────────────────────────────────────

# Path to results folder
base_results_folder = os.path.join(os.path.dirname(__file__), '..', 'results')

# Regexes to extract deltat and lead time from nested folder names
deltat_pattern = re.compile(r'deltat_(\d+)')
lead_pattern = re.compile(r'lead_(\d+)h')

results = []

for variable in os.listdir(base_results_folder):
    variable_dir = os.path.join(base_results_folder, variable)
    if not os.path.isdir(variable_dir):
        continue
    for deltat_folder in os.listdir(variable_dir):
        deltat_match = deltat_pattern.fullmatch(deltat_folder)
        if not deltat_match:
            continue
        deltat = int(deltat_match.group(1))
        deltat_dir = os.path.join(variable_dir, deltat_folder)
        for lead_folder in os.listdir(deltat_dir):
            lead_match = lead_pattern.fullmatch(lead_folder)
            if not lead_match:
                continue
            lead_hours = int(lead_match.group(1))
            best_file = os.path.join(deltat_dir, lead_folder, 'best_trial.txt')
            if not os.path.exists(best_file):
                continue
            metrics = {}
            with open(best_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, _, val = line.strip().partition(':')
                        key = key.strip()
                        val = val.strip()
                        if key.startswith('val_'):
                            try:
                                metrics[key] = ast.literal_eval(val)
                            except (ValueError, SyntaxError):
                                pass
            if METRIC in metrics:
                results.append((deltat, lead_hours, metrics[METRIC]))

# Convert to sorted list
results.sort(key=lambda x: (x[0], x[1]))  # sort by deltat then lead_hours

# Group by deltat
grouped = {}
for deltat, lead_hours, value in results:
    grouped.setdefault(deltat, []).append((lead_hours, value))

# Plot
plt.figure(figsize=(8, 5))
for deltat, values in grouped.items():
    values.sort(key=lambda x: x[0])  # sort by lead_hours
    hours = [v[0] for v in values]
    metric_values = [v[1] for v in values]
    plt.plot(hours, metric_values, marker='o', label=f"Δt = {deltat}h")

plt.xlabel("Lead Time (hours)")
plt.ylabel(METRIC)
plt.title(f"{METRIC} vs Lead Time (per Δt)")
plt.grid(True)
plt.legend(title="Sampling Interval")
plt.tight_layout()
plt.show()
