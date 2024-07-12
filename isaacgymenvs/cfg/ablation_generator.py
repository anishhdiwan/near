import random
import yaml
import os, sys
from datetime import datetime
from pathlib import Path
import argparse
import re

import itertools
from experiment_generator import generate_seeds

FILE_PATH = os.path.dirname(__file__)
sys.path.append(FILE_PATH)

algos = ["HumanoidNEAR"]


motions = [
    "amp_humanoid_walk.yaml", # standard
    # "amp_humanoid_run.yaml",
    "amp_humanoid_crane_pose.yaml", # dynamics learning
    # "amp_humanoid_single_left_punch.yaml",
    # "amp_humanoid_tai_chi.yaml" # long horizon task
]

cfg_settings = [
    [-1, 1.0, True], # default (anneal, style rew, ncsnv2)
    [-1, 0.5, True], # anneal, combined rewards, ncsnv2
    [5, 1.0, True], # no anneal, style rew, ncsnv2
    [5, 0.5, True], # no anneal, combined rew, ncsnv2
    [-1, 1.0, False], # default (anneal, style rew, no ncsnv2)
]

task_specific_cfg = {
    "amp_humanoid_walk.yaml": "headless=True max_iterations=60e6 num_envs=4096 ++train.params.config.minibatch_size=8192",
    "amp_humanoid_run.yaml":"headless=True max_iterations=60e6 num_envs=4096 ++train.params.config.minibatch_size=8192",
    "amp_humanoid_crane_pose.yaml":"headless=True max_iterations=60e6 num_envs=4096 ++train.params.config.minibatch_size=8192",
    "amp_humanoid_single_left_punch.yaml":"headless=True max_iterations=80e6 num_envs=4096 ++train.params.config.minibatch_size=8192",
    "amp_humanoid_tai_chi.yaml":"headless=True max_iterations=100e6 num_envs=4096 ++train.params.config.minibatch_size=8192",
}

near_task_specific_cfg = {
    "amp_humanoid_walk.yaml": "++train.params.config.near_config.training.n_iters=150000",
    "amp_humanoid_run.yaml": "++train.params.config.near_config.training.n_iters=120000",
    "amp_humanoid_crane_pose.yaml": "++train.params.config.near_config.training.n_iters=100000",
    "amp_humanoid_single_left_punch.yaml": "++train.params.config.near_config.training.n_iters=120000",
    "amp_humanoid_tai_chi.yaml": "++train.params.config.near_config.training.n_iters=150000"
}

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
                        w_style = ablation[1]
                        w_task = 1.0 - w_style
                        ncsnv2 = ablation[2]
                        if ncsnv2:
                            ncsn_settings = "++train.params.config.near_config.model.sigma_begin=20 ++train.params.config.near_config.model.L=50 ++train.params.config.near_config.model.ema=True"
                        else:
                            ncsn_settings = "++train.params.config.near_config.model.sigma_begin=10 ++train.params.config.near_config.model.L=10 ++train.params.config.near_config.model.ema=False"

                        cmd = [
f"task={algo} ++task.env.motion_file={motion} seed={seed} \
{task_specific_cfg[motion]} \
++train.params.config.near_config.inference.sigma_level={annealing} \
++train.params.config.near_config.model.ncsnv2={ncsnv2} \
++train.params.config.near_config.inference.task_reward_w={w_task} \
++train.params.config.near_config.inference.energy_reward_w={w_style} \
{ncsn_settings}", 

f"ABLATION_{algo}_{os.path.splitext(motion)[0].replace('amp_humanoid_', '')}_{annealing}_{ncsnv2}_w_style_{str(w_style).replace('.', '')}_{seed}"]
                        pending_cmds.append(cmd)
                        counter += 1

        cmds = [{"algos": algos, "motions":motions, "seeds":seeds, "pending_cmds":pending_cmds, "completed_cmds":[], "num_runs": counter}]
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
        pattern = r"\+\+task\.env\.motion_file=([^ ]+)"
        rgx_match = re.search(pattern, next_cmd[0])
        motion = rgx_match.group(1)
        command_to_pass = next_cmd[0] + f" experiment={next_cmd[1]}" + f" {near_task_specific_cfg[motion]}"

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