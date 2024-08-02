from tbparse import SummaryReader
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from pathlib import Path
import pandas as pd
import random

plt.rcParams['text.usetex'] = True

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
    # event_file = "./ncsn_runs/temp_runs/"
    event_file = "../../NEAR_experiments/plot_runs_temp/"
    
    df_path = Path(os.path.join(PARENT_DIR_PATH, "../../NEAR_experiments/plot_runs_df.pkl"))
    if df_path.is_file():
        df = pd.read_pickle(df_path)
    else:
        reader = SummaryReader(event_file, pivot=True, extra_columns={'dir_name'})
        df = reader.scalars
        df.to_pickle(df_path)
    

    experiment_lables = {
        r"NCSN-v2 $\vert$ Annealing $\vert$ $r^{energy}$":["ABLATION_HumanoidNEAR_walk_-1_True_w_style_10_42", "ABLATION_HumanoidNEAR_walk_-1_True_w_style_10_700", "ABLATION_HumanoidNEAR_walk_-1_True_w_style_10_8125"],
        r"NCSN-v2 $\vert$ Annealing $\vert$ $0.5 r^{energy} + 0.5 r^{task}$":["ABLATION_HumanoidNEAR_walk_-1_True_w_style_05_42", "ABLATION_HumanoidNEAR_walk_-1_True_w_style_05_700", "ABLATION_HumanoidNEAR_walk_-1_True_w_style_05_8125"],
        r"NCSN-v2 $\vert$ $\sigma = 18.64 (lvl. 5)$ $\vert$ $r^{energy}$":["ABLATION_HumanoidNEAR_walk_5_True_w_style_10_42", "ABLATION_HumanoidNEAR_walk_5_True_w_style_10_700", "ABLATION_HumanoidNEAR_walk_5_True_w_style_10_8125"],
        r"NCSN-v2 $\vert$ $\sigma = 18.64 (lvl. 5)$ $\vert$ $0.5 r^{energy} + 0.5 r^{task}$":["ABLATION_HumanoidNEAR_walk_5_True_w_style_05_42", "ABLATION_HumanoidNEAR_walk_5_True_w_style_05_700", "ABLATION_HumanoidNEAR_walk_5_True_w_style_05_8125"],
        r"NCSN-v1 $\vert$ Annealing $\vert$ $r^{energy}$":["ABLATION_HumanoidNEAR_walk_-1_False_w_style_10_42", "ABLATION_HumanoidNEAR_walk_-1_False_w_style_10_700", "ABLATION_HumanoidNEAR_walk_-1_False_w_style_10_8125"],
    }
    # trial_names = ["Humanoid_SM_run0/summaries/_", "Humanoid_SM_run1/summaries/_", "Humanoid_SM_run2/summaries/_"]
    
    scalars = [
        # "episode_lengths/step", 
        "mean_dtw_pose_error/step", 
        "minibatch_combined_reward/step", 
        # "minibatch_energy/step", 
        # "ncsn_perturbation_level/step", 
        "root_body_acceleration/step",
        "root_body_jerk/step",
        "root_body_velocity/step",
        "spectral_arc_length/step",
        ]
    scalar_labels = [
        # "Episode Length", 
        "Average Pose Error", 
        "Horizon Return", 
        # "Horizon Energy Return", 
        # "NCSN Perturbation Level", 
        "Root Body Acceleration",
        "Root Body Jerk",
        "Root Body Velocity",
        "Spectral Arc Length (SPARC)",
        ]
    expert_values = {"spectral_arc_length/step": -5.40, "root_body_velocity/step": 1.31, "root_body_acceleration/step": 3.37, "root_body_jerk/step": 130.11}
    title = "Humanoid Walk"

    for idx, scalar in enumerate(scalars):
        colour_idx = 0
        annotations = {}
        annotate = True
        for exp_label, trial_names in experiment_lables.items():
            trial_names = [name + "/summaries" for name in trial_names]
            exp_dfs = []
            for trial in trial_names:
                exp_dfs.append(df[df['dir_name'] == trial])

            # for idx, scalar in enumerate(scalars):
            exp_scalars, exp_steps = get_scalars_from_dfs(scalar, exp_dfs)
            exp_scalars = np.stack(exp_scalars, axis=1)
            mean_scalar = np.mean(exp_scalars, axis=1)
            std_scalar = np.std(exp_scalars, axis=1)
            std_interval = 1.0
            min_interval = mean_scalar - std_interval*std_scalar
            max_interval = mean_scalar + std_interval*std_scalar

            plt.plot(exp_steps, mean_scalar, color=Colours[colour_idx], linewidth=1.5, label=exp_label)
            plt.fill_between(exp_steps, min_interval, max_interval, alpha=0.2, color=Colours[colour_idx])
            if annotate:
                if scalar not in ["minibatch_combined_reward/step"]:
                    annotations[round(mean_scalar[-1], 4)] = [(exp_steps[-1], mean_scalar[-1]), Colours[colour_idx]]
                    # offset = random.choice([x for x in np.arange(mean_scalar.max()*0., mean_scalar.max()*0.3, mean_scalar.max()*0.1)])
                    # offset = np.arange(mean_scalar.max()*0.1, mean_scalar.max()*0.6, mean_scalar.max()*0.1)[colour_idx]
                    # plt.annotate(f'{round(mean_scalar[-1], 2)}', xy=(exp_steps[-1], mean_scalar[-1]), xytext=(exp_steps[-1] + exp_steps.max()*0.06,  mean_scalar[-1] + offset), arrowprops=dict(arrowstyle='->', color=Colours[colour_idx]))
                
            colour_idx += 1


        plt.xlabel('Training Samples')
        plt.ylabel(scalar_labels[idx])
        plt.title(title)
        plt.tight_layout()
        if scalar in list(expert_values.keys()):
            plt.axhline(y=expert_values[scalar], color='#f032e6', linestyle='-', linewidth=1.5, label="Expert's Value")
            if annotate:
                plt.annotate(f'{expert_values[scalar]}', xy=(plt.gca().get_xlim()[1], expert_values[scalar]), xytext=(1.001*plt.gca().get_xlim()[1], expert_values[scalar]), color='#f032e6')

        if annotate:    
            if len(annotations) > 0:
                for offset_idx, text in enumerate(sorted(list(annotations.keys()), key=float)):
                    offset = float(offset_idx+1)*0.05*plt.gca().get_ylim()[1]
                    plt.annotate(round(text,2), xy=annotations[text][0], xytext=(1.001*plt.gca().get_xlim()[1],  annotations[text][0][1]+offset), arrowprops=dict(arrowstyle='->', color=annotations[text][1]))



        plt.legend()
        plt.show()

