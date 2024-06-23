# from tbparse import SummaryReader
import matplotlib.pyplot as plt
import numpy as np
import os, sys

PARENT_DIR_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(PARENT_DIR_PATH)

# https://sashamaps.net/docs/tools/20-colors/
# Colours = [
#     '#000000',
#     '#e6194B', 
#     '#3cb44b', 
#     # '#ffe119', 
#     '#4363d8', 
#     '#f58231', 
#     '#911eb4', 
#     '#42d4f4', 
#     '#f032e6', 
#     # '#bfef45', 
#     # '#fabed4', 
#     '#469990', 
#     # '#dcbeff', 
#     '#9A6324', 
#     # '#fffac8', 
#     '#800000', 
#     # '#aaffc3', 
#     '#808000', 
#     # '#ffd8b1', 
#     '#000075', 
#     # '#a9a9a9', 
#     # '#ffffff', 
# ]

Colours = [
    '#000000', '#42d4f4', '#e6194B', '#4363d8', '#f58231', '#000075',
    '#3cb44b', '#911eb4', '#469990', '#f032e6', '#9A6324', '#808000',
    '#800000'
]


def get_scalars_from_dfs(scalar, trial_dfs):
    # Returns a list of np  arrays, of the scalar from each trial and a np array of steps
    trial_scalars = []
    trial_steps = None
    for trial_df in trial_dfs:
        series_df = trial_df[[scalar, 'step']]
        series_df = series_df.dropna(subset=[scalar])
        trial_scalars.append(series_df[scalar].to_numpy())
        if trial_steps is None:
            trial_steps = series_df['step'].to_numpy()

    return trial_scalars, trial_steps


if __name__ == "__main__":

    # Get all events in this dir
    event_file = "./ncsn_runs/temp_runs/"
    reader = SummaryReader(event_file, pivot=True, extra_columns={'dir_name'})
    df = reader.scalars


    trial_names = ["Humanoid_SM_run0/summaries/_", "Humanoid_SM_run1/summaries/_", "Humanoid_SM_run2/summaries/_"]
    scalars = ["loss", "demo_data_energy/sigma_level_9"]

    exp_dfs = []
    for trial in trial_names:
        exp_dfs.append(df[df['dir_name'] == trial])


    for idx, scalar in enumerate(scalars):
        exp_scalars, exp_steps = get_scalars_from_dfs(scalar, exp_dfs)
        exp_scalars = np.stack(exp_scalars, axis=1)
        mean_scalar = np.mean(exp_scalars, axis=1)
        std_scalar = np.std(exp_scalars, axis=1)
        std_interval = 1.0
        min_interval = mean_scalar - std_interval*std_scalar
        max_interval = mean_scalar + std_interval*std_scalar

        plt.plot(exp_steps, mean_scalar, color=Colours[idx], linewidth=1)
        plt.fill_between(exp_steps, min_interval, max_interval, alpha=0.2, color=Colours[idx])

        plt.xlabel('steps')
        plt.ylabel(scalar)
        plt.show()

