from tbparse import SummaryReader
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os, sys

PARENT_DIR_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(PARENT_DIR_PATH)

# https://sashamaps.net/docs/tools/20-colors/
Colours = [
	'#800000',  # Maroon (99.99%)
	'#4363d8',  # Blue (99.99%)
	'#ffe119',  # Yellow (99.99%)
	'#e6beff',  # Lavender (99.99%)
	'#f58231',  # Orange (99.99%)
	'#3cb44b',  # Green (99%)
	'#000075',  # Navy (99.99%)
	'#e6194b',  # Red (99%)
	'#46f0f0',  # Cyan (99%)
	'#f032e6',  # Magenta (99%)
	'#9a6324',  # Brown (99%)
	'#008080',  # Teal (99%)
	'#911eb4',  # Purple (95%*)
	'#aaffc3',  # Mint (99%)
	'#ffd8b1',  # Apiroct (95%)
	'#bcf60c',  # Lime (95%)
	'#fabed4',  # Pink (99%)
	'#808000',  # Olive (95%)
	'#fffac8',  # Beige (99%)
	#'#a9a9a9',
	#'#ffffff',
	#'#000000'
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

