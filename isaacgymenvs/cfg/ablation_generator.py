import random
import yaml
import os, sys
from datetime import datetime
from pathlib import Path
import argparse

import itertools
from experiment_generator import generate_seeds

FILE_PATH = os.path.dirname(__file__)
sys.path.append(FILE_PATH)

algos = ["HumanoidNEAR"]

motions = [
    "amp_humanoid_walk.yaml", # standard 
    "amp_humanoid_cartwheel.yaml", # dynamics learning
    "amp_humanoid_martial_arts_bassai.yaml" # long horizon task
]

cfg_settings = [
    [-1, True, 1.0], # default (anneal, temporal, style rew)
    [-1, True, 0.5], # combined rewards
    [-1, False, 1.0], # no teporal
    [5, True, 1.0], # no anneal
    [5, False, 1.0], # no anneal, no temporal
    [-1, False, 0.5], # anneal, no temporal, combined
]

manual_seeds = []


def generate_train_commands():
    train_cmds = Path(os.path.join(FILE_PATH, "ablation_cmds.yaml"))
    if train_cmds.is_file():
        # Avoid overwriting automatically
        pass

    else:

        seeds = generate_seeds(manual_seeds=manual_seeds)

        pending_cmds = []
        counter = 0
        for algo in algos:
            for motion in motions:
                for seed in seeds:
                    for ablation in cfg_settings:
                        annealing = ablation[0]
                        temporal_feature = ablation[1]
                        w_style = ablation[2]
                        w_task = 1.0 - w_style
                        cmd = [
f"task={algo} ++task.env.motion_file={motion} seed={seed} \
++train.params.config.near_config.inference.sigma_level={annealing} \
++train.params.config.near_config.model.encode_temporal_feature={temporal_feature} \
++train.params.config.near_config.inference.task_reward_w={w_task} \
++train.params.config.near_config.inference.energy_reward_w={w_style}", 

f"{algo}_{os.path.splitext(motion)[0].replace('amp_humanoid_', '')}_{annealing}_{temporal_feature}_w_style_{str(w_style).replace('.', '')}_{seed}"]
                        pending_cmds.append(cmd)
                        counter += 1

        cmds = [{"algos": algos, "motions":motions, "seeds":seeds, "pending_cmds":pending_cmds, "completed_cmds":[]}]
        with open(os.path.join(FILE_PATH, "ablation_cmds.yaml"), 'w') as yaml_file:
            yaml.dump(cmds, yaml_file, default_flow_style=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="type of model to train (reinforcement learning or NCSN)")
    args = parser.parse_args()

    # Does nothing if cmds already exist
    generate_train_commands()

    # open the training commands yaml file
    with open(os.path.join(FILE_PATH, "ablation_cmds.yaml")) as stream:
        try:
            cmds = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


    if args.model == "ncsn":
        try:
            # get the first one
            next_cmd = cmds[0]['pending_cmds'][0]
        except IndexError as e:
            print("done")
            quit()

        # add experiment name to cmd and add it to the completed cmds
        command_to_pass = next_cmd[0] + f" experiment={next_cmd[1]}"

        cmds[0]['completed_cmds'].append([command_to_pass])
        # save the training commands yaml file
        with open(os.path.join(FILE_PATH, "ablation_cmds.yaml"), 'w') as yaml_file:
            yaml.dump(cmds, yaml_file, default_flow_style=False) 

        print(command_to_pass)
            
    elif args.model == "rl":

        try:
            # pop the first one
            next_cmd = cmds[0]['pending_cmds'].pop(0)
        except IndexError as e:
            print("done")
            quit()

        # pass ncsn checkpoint as well
        # add experiment name to cmd and add it to the completed cmds
        ncsn_dir = next_cmd[1]
        eb_model_checkpoint = f"ncsn_runs/{ncsn_dir}/nn/checkpoint.pth"
        running_mean_std_checkpoint = f"ncsn_runs/{ncsn_dir}/nn/running_mean_std.pth"
        command_to_pass = next_cmd[0] + f" ++train.params.config.near_config.inference.eb_model_checkpoint={eb_model_checkpoint}" \
        + f" ++train.params.config.near_config.inference.running_mean_std_checkpoint={running_mean_std_checkpoint}" + f" experiment={next_cmd[1]}"
        
        cmds[0]['completed_cmds'][-1].append(command_to_pass)


        # save the training commands yaml file
        with open(os.path.join(FILE_PATH, "ablation_cmds.yaml"), 'w') as yaml_file:
            yaml.dump(cmds, yaml_file, default_flow_style=False) 

        print(command_to_pass)