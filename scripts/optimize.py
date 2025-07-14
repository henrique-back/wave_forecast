import os
import optuna
import optuna.visualization as vis
from utils import get_freqs, set_seed, data_processing, save_progress
from nn import objective
from functools import partial


# Set randomness seed
set_seed(42)

# Set parameters
lead_times = [1, 6, 12, 24]
target = 'density'
n_trials = 20

# Process data
folder_path = 'buoy_data'
density, alpha_1, alpha_2, r_1 = data_processing(folder_path, save_path='buoy_data\processed_data.pkl')
freqs = get_freqs(density)

for lead_time in lead_times:
    print(f"\n=== Optimizing for lead time = {lead_time} ===")

    # Set objective function from objective defined in nn\optimization.py
    objective_fn = partial(objective,
                        density=density,
                        alpha_1=alpha_1,
                        alpha_2=alpha_2,
                        r_1=r_1, 
                        freqs=freqs, 
                        lead_time=lead_time, 
                        target=target) 

    results_folder = os.path.join(os.path.dirname(__file__), '..', 'results', f'lead_time_{lead_time}')
    os.makedirs(results_folder, exist_ok=True)

    # Run optuna
    study = optuna.create_study(direction='minimize')  # minimize val loss
    study.optimize(objective_fn, n_trials=n_trials,
                   callbacks=[lambda study, trial: save_progress(study, trial, results_folder)])
    print("Best trial:")
    print(study.best_trial.params)
    print("Validation loss:", study.best_value)

    # Save results to results folder
    result_file = os.path.join(results_folder, 'best_trial.txt')
    with open(result_file, 'w') as f:
        f.write("Best trial parameters:\n")
        f.write(str(study.best_trial.params) + '\n')
        f.write(f"Validation loss: {study.best_value}\n")

    print(f"Results saved to {result_file}")

    fig = vis.plot_param_importances(study)
    fig.write_html(os.path.join(results_folder, 'param_importances.html'))

    fig = vis.plot_optimization_history(study)
    fig.write_html(os.path.join(results_folder, 'optimization_history.html'))

    print(f"Visualizations saved to {results_folder}")