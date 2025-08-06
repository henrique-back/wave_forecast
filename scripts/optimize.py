print("Importing packages")
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
import pandas as pd
import optuna
import optuna.visualization as vis
from utils import get_freqs, set_seed, data_processing, save_progress
from nn import objective
from functools import partial

print("Current working directory:", os.getcwd())

# Set randomness seed
set_seed(42)

# Set parameters
lead_times = [1, 6, 12, 24]
target = 'hs'
n_trials = 100

# Process data
project_root = Path(__file__).resolve().parent.parent
folder_path = project_root / "buoy_data"
file_path = folder_path / "processed_data.pkl"

# Load from file if it exists
if file_path.exists():
    dfs_interpolated = pd.read_pickle(file_path)
    density, alpha_1, alpha_2, r_1 = dfs_interpolated
    print("Loaded preprocessed wave spectral data")
else:
    from utils.data_processing import data_processing  # or wherever your function lives
    density, alpha_1, alpha_2, r_1 = data_processing(folder_path, save_path=file_path)

freqs = get_freqs(density)

deltats = [1, 6, 12]
for deltat in deltats:
    # Downsample
    density_d = density[::deltat]
    alpha_1_d = alpha_1[::deltat]
    alpha_2_d = alpha_2[::deltat]
    r_1_d = r_1[::deltat]

    for lead_time in lead_times:
        # Convert lead_time to hours
        lead_time_hours = lead_time * deltat
        print(f"\n=== Optimizing for deltat = {deltat}, lead time = {lead_time_hours}h ===")

        # Define objective function
        objective_fn = partial(
            objective,
            density=density_d,
            alpha_1=alpha_1_d,
            alpha_2=alpha_2_d,
            r_1=r_1_d, 
            freqs=freqs, 
            lead_time=lead_time,
            target=target) 

        # Folder for results
        results_folder = os.path.join(
            os.path.dirname(__file__), '..', 'results', f'deltat_{deltat}_lead_{lead_time_hours}h'
        )
        os.makedirs(results_folder, exist_ok=True)

        # Create unique Optuna study name for this combo
        study_name = f'hs_wave_transformer_deltat_{deltat}_lead_{lead_time_hours}h'

        # Run optuna
        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///optuna_study.db",
            direction='minimize',
            load_if_exists=True)
        
        study.optimize(objective_fn, n_trials=n_trials,
                    callbacks=[lambda study, trial: save_progress(study, trial, results_folder)])
        print("Best trial:")
        print(study.best_trial.params)
        print("Validation loss:", study.best_value)

        result_file = os.path.join(results_folder, 'best_trial.txt')
        with open(result_file, 'w') as f:
            f.write(f"Delta t: {deltat}\n")
            f.write(f"Lead time (steps): {lead_time}\n")
            f.write(f"Lead time (hours): {lead_time_hours}\n")
            f.write("Best trial parameters:\n")
            f.write(str(study.best_trial.params) + '\n')
            f.write(f"Validation loss: {study.best_value}\n")

        print(f"Results saved to {result_file}")

        # Save visualizations
        fig = vis.plot_param_importances(study)
        fig.write_html(os.path.join(results_folder, 'param_importances.html'))

        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(results_folder, 'optimization_history.html'))

        print(f"Visualizations saved to {results_folder}")