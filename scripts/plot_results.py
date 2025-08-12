import os
import re
import matplotlib.pyplot as plt

# Path to results folder
base_results_folder = os.path.join(os.path.dirname(__file__), '..', 'results')

# Regex to extract deltat and lead time in hours from folder name
pattern = re.compile(r'hs_wave_transformer_deltat_(\d+)_lead_(\d+)h')

results = []

for folder in os.listdir(base_results_folder):
    match = pattern.match(folder)
    if match:
        deltat = int(match.group(1))
        lead_hours = int(match.group(2))

        best_file = os.path.join(base_results_folder, folder, 'best_trial.txt')
        if os.path.exists(best_file):
            with open(best_file, 'r') as f:
                lines = f.readlines()
                # Find line with Validation loss
                for line in lines:
                    if "Validation loss" in line:
                        val_loss = float(line.strip().split(":")[1])
                        results.append((deltat, lead_hours, val_loss))
                        break

# Convert to sorted list
results.sort(key=lambda x: (x[0], x[1]))  # sort by deltat then lead_hours

# Group by deltat
grouped = {}
for deltat, lead_hours, val_loss in results:
    if deltat not in grouped:
        grouped[deltat] = []
    grouped[deltat].append((lead_hours, val_loss))

# Plot
plt.figure(figsize=(8, 5))
for deltat, values in grouped.items():
    values.sort(key=lambda x: x[0])  # sort by lead_hours
    hours = [v[0] for v in values]
    losses = [v[1] for v in values]
    plt.plot(hours, losses, marker='o', label=f"Δt = {deltat}h")

plt.xlabel("Lead Time (hours)")
plt.ylabel("Best Validation Loss (RMSE)")
plt.title("Best Forecast Error vs Lead Time (per Δt)")
plt.grid(True)
plt.legend(title="Sampling Interval")
plt.tight_layout()
plt.savefig('my_plot.png')