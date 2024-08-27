from tbparse import SummaryReader
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

plt.rcParams['text.usetex'] = True
# plt.rcParams.update({'font.size': 22})

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

PLOT_AVG = False

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
    event_file = "../../NEAR_experiments/plot_runs_temp/"
    reader = SummaryReader(event_file, pivot=True, extra_columns={'dir_name'})
    df = reader.scalars


    trial_names = ["DISCEXP_HumanoidAMP_run_700_500k/summaries", "DISCEXP_HumanoidAMP_run_700_2M/summaries", "DISCEXP_HumanoidAMP_run_700_5M/summaries"]
    trial_labels = ["After 0.5e6 samples", "After 2e6 samples", "After 5e6 samples"]
    scalars = ["disc_experiment/disc_combined_acc/iter", "disc_experiment/disc_loss_least_sq/iter", "disc_experiment/grad_disc_obs/iter"]
    # xlabels = ["Training Iterations"]
    titles = [r"Discriminator Accuracy (on both $p_G$ and $p_D$)", r"Discriminator Error", r"Grad. Disc()"]
    ylabels = ["Accuracy", "Cross-Entropy", r"$\left\lVert(\nabla_x D(x))\right\rVert_2$"]

    fontsize = 16
    plt.rcParams.update({'font.size': fontsize})
    # params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
    # plt.rcParams.update(params)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    exp_dfs = []
    for trial in trial_names:
        exp_dfs.append(df[df['dir_name'] == trial])


    for idx, scalar in enumerate(scalars):
        exp_scalars, exp_steps = get_scalars_from_dfs(scalar, exp_dfs)

        if PLOT_AVG:

            if scalar != "disc_experiment/disc_combined_acc/iter":
                plt.yscale("log")
            else:
                plt.yscale("linear")

            exp_scalars = np.stack(exp_scalars, axis=1)
            mean_scalar = np.mean(exp_scalars, axis=1)
            std_scalar = np.std(exp_scalars, axis=1)
            std_interval = 1.0
            min_interval = mean_scalar - std_interval*std_scalar
            max_interval = mean_scalar + std_interval*std_scalar

            plt.plot(exp_steps, mean_scalar, color=Colours[idx], linewidth=1)
            plt.fill_between(exp_steps, min_interval, max_interval, alpha=0.2, color=Colours[idx])

            plt.xlabel('Training Iterations')
            plt.ylabel(ylabels[idx])
            plt.show()
        
        else:
            if scalar != "disc_experiment/disc_combined_acc/iter":
                plt.yscale("log")
            else:
                plt.yscale("linear")

            for exp_idx, exp_scalar in enumerate(exp_scalars):
                plt.plot(exp_steps, exp_scalar, color=Colours[exp_idx], linewidth=1, label=trial_labels[exp_idx])


            plt.xlabel('Training Iterations')
            plt.ylabel(ylabels[idx])
            plt.title(titles[idx])
            if scalar == "disc_experiment/disc_combined_acc/iter":
                plt.legend(loc=(0.08, 0.05))
            else:
                plt.legend()

            if scalar == "disc_experiment/disc_combined_acc/iter":
                x1 = 0.
                x2 = 150.
                y1 = 0.8
                y2 = 1.02
                ax = plt.gca()
                axins = zoomed_inset_axes(ax, 2.5, loc=4) # zoom = 2.5
                for exp_idx, exp_scalar in enumerate(exp_scalars):
                    axins.plot(exp_steps, exp_scalar, color=Colours[exp_idx], linewidth=1)
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                plt.xticks(visible=False)
                plt.yticks(visible=False)
                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
                plt.draw()
            
            plt.tight_layout()
            plt.show()


