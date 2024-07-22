import matplotlib.pyplot as plt
import numpy as np
import os, sys
from os.path import isfile, join
from experiment_plotter import Colours
import pickle

plt.rcParams['text.usetex'] = True

PARENT_DIR_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(PARENT_DIR_PATH)
iters_per_epoch = int((6 * 16 * 1024)/2048)

def get_saved_scalars(events_dir, trial_names):
    trial_scalars = []
    for trial_name in trial_names:
        trial_path = os.path.join(events_dir, trial_name)
        pkl_file = [f for f in os.listdir(trial_path) if isfile(join(trial_path, f)) and f.endswith(".pkl")][0]
        pkl_file = os.path.join(events_dir, trial_name, pkl_file)
        with open(pkl_file, 'rb') as handle:
            scalar_dict = pickle.load(handle)
        trial_scalars.append(scalar_dict)
    
    trial_steps = iters_per_epoch * np.arange(len(trial_scalars[0]['disc_expt_mean_rew']))

    keys = list(trial_scalars[0].keys())
    inverted_dict = {}
    for key in keys:
        for trial_scalar_dict in trial_scalars:
            if key not in list(inverted_dict.keys()):
                inverted_dict[key] = []
            inverted_dict[key].append(trial_scalar_dict[key])

    return inverted_dict, trial_steps


if __name__ == "__main__":
    events_dir = "../../NEAR_experiments/plot_runs_temp/"
    trial_names = ["DISCEXP_HumanoidAMP_run_700_500k/nn", "DISCEXP_HumanoidAMP_run_700_2M/nn", "DISCEXP_HumanoidAMP_run_700_5M/nn"]
    trial_labels = ["After 0.5e6 samples", "After 2e6 samples", "After 5e6 samples"]
    scalars = [['disc_expt_mean_pred', 'disc_expt_std_pred'], ['disc_expt_mean_rew', 'disc_expt_std_rew']]
    titles = ["Discriminator Predictions", "Discriminator Reward"]
    ylabels = [r"$D_{\theta_D}(W(\pi_{\theta_G}(s)))$", r"$rew\_fn(D_{\theta_D}(W(\pi_{\theta_G}(s))))$"]

    std_interval = 1.0

    exp_scalars, exp_steps = get_saved_scalars(events_dir, trial_names)

    for ind, scalar in enumerate(scalars):
        mean_key = scalar[0]
        std_key = scalar[1]
        mean_scalars = exp_scalars[mean_key]
        std_scalars = exp_scalars[std_key]

        for idx, exp_scalar in enumerate(mean_scalars):
            min_exp_scalar = exp_scalar - std_interval*std_scalars[idx]
            max_exp_scalar = exp_scalar + std_interval*std_scalars[idx]

            plt.plot(exp_steps, exp_scalar, color=Colours[idx], linewidth=1, label=trial_labels[idx])
            plt.fill_between(exp_steps, min_exp_scalar, max_exp_scalar, alpha=0.2, color=Colours[idx])

        plt.xlabel('Training Iterations')
        plt.ylabel(ylabels[ind])
        plt.title(titles[ind])
        plt.legend()
        plt.tight_layout()
        plt.show()




