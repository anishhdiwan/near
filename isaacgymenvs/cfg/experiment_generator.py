import random
import yaml
import os, sys
from datetime import datetime
from pathlib import Path
import argparse
import re

FILE_PATH = os.path.dirname(__file__)
sys.path.append(FILE_PATH)

algos = [
    "HumanoidNEAR", 
    "HumanoidAMP"
]

motions = [
    "amp_humanoid_walk.yaml",
    "amp_humanoid_run.yaml",
    "amp_humanoid_crane_pose.yaml",
    "amp_humanoid_single_left_punch.yaml",
    # "amp_humanoid_tai_chi.yaml"
]

task_specific_cfg = {
    "amp_humanoid_walk.yaml": "headless=True max_iterations=60e6 num_envs=4096 ++train.params.config.minibatch_size=32768",
    "amp_humanoid_run.yaml":"headless=True max_iterations=60e6 num_envs=4096 ++train.params.config.minibatch_size=32768",
    "amp_humanoid_crane_pose.yaml":"headless=True max_iterations=60e6 num_envs=4096 ++train.params.config.minibatch_size=32768",
    "amp_humanoid_single_left_punch.yaml":"headless=True max_iterations=80e6 num_envs=4096 ++train.params.config.minibatch_size=32768",
    "amp_humanoid_tai_chi.yaml":"headless=True max_iterations=100e6 num_envs=4096 ++train.params.config.minibatch_size=32768",
}

near_task_specific_cfg = {
    "amp_humanoid_walk.yaml": "++train.params.config.near_config.training.n_iters=150000",
    "amp_humanoid_run.yaml": "++train.params.config.near_config.training.n_iters=120000",
    "amp_humanoid_crane_pose.yaml": "++train.params.config.near_config.training.n_iters=100000",
    "amp_humanoid_single_left_punch.yaml": "++train.params.config.near_config.training.n_iters=120000",
    "amp_humanoid_tai_chi.yaml": "++train.params.config.near_config.training.n_iters=150000"
}

amp_task_specific_cfg = {
    "amp_humanoid_walk.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_run.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_crane_pose.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_single_left_punch.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_tai_chi.yaml": "++train.params.config.amp_minibatch_size=4096"
}


manual_seeds = []

def generate_seeds(start=0, end=int(1e4), k=3, manual_seeds=[]):

    if manual_seeds != []:
        seeds = manual_seeds
    else:
        seeds = random.sample(range(start, end), k)

    return seeds

def generate_train_commands():
    train_cmds = Path(os.path.join(FILE_PATH, "train_cmds.yaml"))
    if train_cmds.is_file():
        # Avoid overwriting automatically
        pass

    else:

        seeds = generate_seeds(manual_seeds=manual_seeds)
        # print(f"algos {algos}")
        # print(f"motions {motions}")
        # print(f"seeds {seeds}")

        pending_cmds = []
        counter = 0
        for algo in algos:
            for motion in motions:
                for seed in seeds:
                    # cmd = {counter: {"cmd":f"task={algo} ++task.env.motion_file={motion} seed={seed}", "exp_name":f"{algo}_{os.path.splitext(motion)[0].replace('amp_humanoid_', '')}_{seed}"}}
                    cmd = [f"task={algo} ++task.env.motion_file={motion} seed={seed} {task_specific_cfg[motion]}", f"{algo}_{os.path.splitext(motion)[0].replace('amp_humanoid_', '')}_{seed}"]
                    pending_cmds.append(cmd)
                    counter += 1
        
        cmds = [{"algos": algos, "motions":motions, "seeds":seeds, "pending_cmds":pending_cmds, "completed_cmds":[], "num_runs": counter}]
        with open(os.path.join(FILE_PATH, "train_cmds.yaml"), 'w') as yaml_file:
            yaml.dump(cmds, yaml_file, default_flow_style=False)

        # print(f"Train commands generated! Please manually delete the file {train_cmds} to create a new one and call this script again to start training")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="type of model to train (reinforcement learning or NCSN)")
    args = parser.parse_args()

    # Does nothing if cmds already exist
    generate_train_commands()

    # open the training commands yaml file
    with open(os.path.join(FILE_PATH, "train_cmds.yaml")) as stream:
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

        if 'HumanoidNEAR' in next_cmd[0]:
            # add experiment name to cmd and add it to the completed cmds
            pattern = r"\+\+task\.env\.motion_file=([^ ]+)"
            rgx_match = re.search(pattern, next_cmd[0])
            motion = rgx_match.group(1)
            command_to_pass = next_cmd[0] + f" experiment={next_cmd[1]}" + f" {near_task_specific_cfg[motion]}"

            cmds[0]['completed_cmds'].append([command_to_pass])
            # save the training commands yaml file
            with open(os.path.join(FILE_PATH, "train_cmds.yaml"), 'w') as yaml_file:
                yaml.dump(cmds, yaml_file, default_flow_style=False) 

            print(command_to_pass)
            
        elif 'HumanoidAMP' in next_cmd[0]:
            print("") 

    elif args.model == "rl":

        try:
            # pop the first one
            next_cmd = cmds[0]['pending_cmds'].pop(0)
        except IndexError as e:
            print("done")
            quit()

        if 'HumanoidNEAR' in next_cmd[0]:
            # pass ncsn checkpoint as well
            # add experiment name to cmd and add it to the completed cmds
            ncsn_dir = next_cmd[1]
            eb_model_checkpoint = f"ncsn_runs/{ncsn_dir}/nn/checkpoint.pth"
            running_mean_std_checkpoint = f"ncsn_runs/{ncsn_dir}/nn/running_mean_std.pth"
            command_to_pass = next_cmd[0] + f" ++train.params.config.near_config.inference.eb_model_checkpoint={eb_model_checkpoint}" \
            + f" ++train.params.config.near_config.inference.running_mean_std_checkpoint={running_mean_std_checkpoint}" + f" experiment={next_cmd[1]}"
            
            cmds[0]['completed_cmds'][-1].append(command_to_pass)

        elif 'HumanoidAMP' in next_cmd[0]:
            pattern = r"\+\+task\.env\.motion_file=([^ ]+)"
            rgx_match = re.search(pattern, next_cmd[0])
            motion = rgx_match.group(1)
            # add experiment name to cmd and add it to the completed cmds
            command_to_pass = next_cmd[0] + f" experiment={next_cmd[1]}" + f" {amp_task_specific_cfg[motion]}"
            cmds[0]['completed_cmds'].append([command_to_pass])

        # save the training commands yaml file
        with open(os.path.join(FILE_PATH, "train_cmds.yaml"), 'w') as yaml_file:
            yaml.dump(cmds, yaml_file, default_flow_style=False) 

        print(command_to_pass)




    



 


