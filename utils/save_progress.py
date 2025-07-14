import os
import optuna.visualization as vis
import joblib

def save_progress(study, trial, results_folder):
    print(f"Trial {trial.number} finished with value: {trial.value} and params: {trial.params}")
    
    # Save intermediate optimization history
    fig = vis.plot_optimization_history(study)
    fig.write_html(os.path.join(results_folder, f'optimization_history_trial.html'))
    
    # Save current best result
    with open(os.path.join(results_folder, 'current_best.txt'), 'w') as f:
        f.write(f"Best value: {study.best_value}\n")
        f.write(f"Best params: {study.best_trial.params}\n")

        # Save the study object (checkpoint)
    joblib.dump(study, os.path.join(results_folder, 'study.pkl'))
