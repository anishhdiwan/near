import matplotlib.pyplot as plt
import numpy as np
import os, sys
from experiment_plotter import Colours
import argparse
from pathlib import Path
import yaml
import re
import pickle

FILE_PATH = os.path.join(os.path.dirname(__file__))
sys.path.append(FILE_PATH)

motions = {
    "walk": "amp_humanoid_walk.yaml",
    "run": "amp_humanoid_run.yaml",
    "crane_pose": "amp_humanoid_crane_pose.yaml",
    "cartwheel": "amp_humanoid_cartwheel.yaml",
    "jump_in_place": "amp_humanoid_jump_in_place.yaml",
    "bassai": "amp_humanoid_martial_arts_bassai.yaml"
}


def generate_play_commands(algo, checkpoints, trials):
    play_cmds = Path(os.path.join(FILE_PATH, f"{algo}_play_cmds.yaml"))
    if play_cmds.is_file():
        # Avoid overwriting automatically
        pass

    else:
        pending_cmds = []
        for checkpoint in checkpoints:
            for trial in trials:

                trial_motion = trial.replace("HumanoidNEAR_", "")
                trial_motion = trial_motion.replace("HumanoidAMP_", "")
                for motion in list(motions.keys()):
                    if motion in trial_motion:
                        trial_motion_file = motions[motion]

                if algo == "AMP":
                    checkpoints_pth = os.path.join(FILE_PATH, f"../runs/{trial}/nn")
                    trial_checkpoints = [f for f in os.listdir(checkpoints_pth) if os.path.isfile(os.path.join(checkpoints_pth, f))]

                    # Define the regex pattern to match the known checkpoint
                    pattern = rf'_{checkpoint}\.pth$'
                    for filename in trial_checkpoints:
                        if re.search(pattern, filename):
                            pending_cmds.append(f"train.py task=Humanoid{algo} test=True checkpoint=runs/{trial}/nn/{filename} ++task.env.motion_file={trial_motion_file}")

                elif algo=="NEAR":
                    pending_cmds.append(f"train_ncsn.py task=Humanoid{algo} test=True \
++train.params.config.near_config.inference.eb_model_checkpoint=ncsn_runs/{trial}/nn/checkpoint.pth \
++train.params.config.near_config.inference.running_mean_std_checkpoint=ncsn_runs/{trial}/nn/running_mean_std.pth \
++task.env.motion_file={trial_motion_file}")


        cmds = [{"pending_cmds":pending_cmds}]
        with open(os.path.join(FILE_PATH, f"{algo}_play_cmds.yaml"), 'w') as yaml_file:
            yaml.dump(cmds, yaml_file, default_flow_style=False)

def pass_play_commands(args):

    # open the training commands yaml file
    with open(os.path.join(FILE_PATH, f"{args.algo}_play_cmds.yaml")) as stream:
        try:
            cmds = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    try:
        # pop the first one
        command_to_pass = cmds[0]['pending_cmds'].pop(0)
    except IndexError as e:
        print("done")
        plot_series(args)
        quit()
    
    # save the training commands yaml file
    with open(os.path.join(FILE_PATH, f"{args.algo}_play_cmds.yaml"), 'w') as yaml_file:
        yaml.dump(cmds, yaml_file, default_flow_style=False) 

    print(command_to_pass)

def plot_series(args):
    checkpoints = args.checkpoints
    algo = args.algo
    trials = args.trials
    aggregated_data = {}

    for checkpoint in checkpoints:
        checkpoint_data = []
        for trial in trials:
            if algo == "AMP":
                checkpoints_pth = os.path.join(FILE_PATH, f"../runs/{trial}/nn")
                trial_checkpoints = [f for f in os.listdir(checkpoints_pth) if os.path.isfile(os.path.join(checkpoints_pth, f))]

                # Define the regex pattern to match the known checkpoint
                pattern = rf'_{checkpoint}\.pth$'
                for filename in trial_checkpoints:
                    if re.search(pattern, filename):
                        datafile = os.path.splitext(os.path.join(checkpoints_pth, filename))[0] + '_learnt_fn.pkl'

                        with open(datafile, 'rb') as handle:
                            checkpoint_data.append(pickle.load(handle))
        

            elif algo=="NEAR":
                datafile = os.path.join(FILE_PATH, f"../ncsn_runs/{trial}/nn/checkpoint_learnt_fn.pkl")

                with open(datafile, 'rb') as handle:
                    checkpoint_data.append(pickle.load(handle))

        if algo == "AMP":
            checkpoint_aggregated_data = aggregate_checkpoint_data(checkpoint_data)
            # aggregated_data[f"{checkpoint}_steps"] = checkpoint_aggregated_data
            for data_key in list(checkpoint_aggregated_data.keys()):
                if not data_key in aggregated_data.keys():
                    aggregated_data[data_key] = {f"{checkpoint}_steps": checkpoint_aggregated_data[data_key]}
                else:
                    aggregated_data[data_key][f"{checkpoint}_steps"] = checkpoint_aggregated_data[data_key]



    if algo == "NEAR":
        aggregated_data = aggregate_checkpoint_data(checkpoint_data)
    
    plot_checkpoint_data(aggregated_data, algo=algo)


def combine_dicts(*dicts):
    # Get all unique keys from all dictionaries
    all_keys = set().union(*dicts)
    # Create the new dictionary with lists of values
    return {k: [d.get(k) for d in dicts] for k in all_keys}

def stack_sub_dicts(dicts):
    for key in list(dicts.keys()):
        dicts[key] = np.stack(dicts[key], axis=1)

    return dicts

def aggregate_checkpoint_data(checkpoint_data):
    data_keys = list(checkpoint_data[0].keys())
    aggregated_data = {}
    for data_key in data_keys:
        key_data = []
        dicts_list=[]
        for trial_data in checkpoint_data:
            if isinstance(trial_data[data_key], np.ndarray):
                key_data.append(trial_data[data_key])

            elif isinstance(trial_data[data_key], dict):
                dicts_list.append(trial_data[data_key])

        if dicts_list == []:
            key_data = np.stack(key_data, axis=1)
        elif key_data == []:
            combined_data = combine_dicts(*dicts_list)
            key_data = stack_sub_dicts(combined_data)

        aggregated_data[data_key] = key_data
    
    return aggregated_data

def plot_checkpoint_data(aggregated_data, algo):

    if isinstance(aggregated_data['max_sample_perturbation'], np.ndarray):
        data_x = aggregated_data['max_sample_perturbation'][:,0]
        aggregated_data.pop('max_sample_perturbation')
    elif isinstance(aggregated_data['max_sample_perturbation'], dict):
        data_x = next(iter(aggregated_data['max_sample_perturbation'].values()))[:,0]
        aggregated_data.pop('max_sample_perturbation')
    

    data_keys = list(aggregated_data.keys())

    for idx, data_key in enumerate(data_keys):
        data = aggregated_data[data_key]
        if isinstance(data, np.ndarray):
            scalar = data
            mean_scalar = np.mean(scalar, axis=1)
            std_scalar = np.std(scalar, axis=1)
            std_interval = 1.0
            min_interval = mean_scalar - std_interval*std_scalar
            max_interval = mean_scalar + std_interval*std_scalar

            plt.figure(figsize=(8, 6))
            plt.plot(data_x, mean_scalar, color=Colours[idx], linewidth=1.5)
            plt.fill_between(data_x, min_interval, max_interval, alpha=0.1, color=Colours[idx])

            plt.xlabel("max perturbation r (where sample = sample + unif[-r,r])")
            plt.ylabel(data_key)
            plt.title(f"Avg {data_key} vs distance from demo data")
            plt.show()
        elif isinstance(data, dict):
            plt.figure(figsize=(8, 6))
            for ind,k in enumerate(list(data.keys())):
                scalar = data[k]
                mean_scalar = np.mean(scalar, axis=1)
                std_scalar = np.std(scalar, axis=1)
                std_interval = 1.0
                min_interval = mean_scalar - std_interval*std_scalar
                max_interval = mean_scalar + std_interval*std_scalar

                if ind%2 == 0:
                    linestyle = 'solid'
                    hatch = None
                else:
                    linestyle = 'dotted'
                    hatch = None

                if algo=="NEAR":
                    plt.plot(data_x, np.flip(mean_scalar, axis=0), color=Colours[ind], linewidth=1.5, label=f"{data_key}_{k}")
                    plt.fill_between(data_x, np.flip(min_interval, axis=0), np.flip(max_interval, axis=0), alpha=0.1, color=Colours[ind], hatch=hatch, edgecolor=Colours[ind])
                elif algo=="AMP":
                    plt.plot(data_x, mean_scalar, color=Colours[ind], linewidth=1.5, label=f"{data_key}_{k}")
                    plt.fill_between(data_x, min_interval, max_interval, alpha=0.1, color=Colours[ind], hatch=hatch, edgecolor=Colours[ind])

            plt.xlabel("max perturbation r (where sample = sample + unif[-r,r])")
            plt.ylabel(data_key)
            plt.title(f"Avg {data_key} vs distance from demo data")
            plt.legend()
            plt.show()
                






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algo", type=str, required=True, help="whether to plot AMP rewards or NEAR rewards")
    parser.add_argument('-t', '--trials', nargs='+', required=True, help='the experiment trials to plot')
    parser.add_argument('-c', '--checkpoints', nargs='+', required=True, help='the checkpoints to plot')
    args = parser.parse_args()

    # Generate commands yaml
    generate_play_commands(args.algo, args.checkpoints, args.trials)
    # Pass a command everytime this is called
    pass_play_commands(args)








