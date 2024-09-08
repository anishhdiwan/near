import random
import yaml
import os, sys
from datetime import datetime
from pathlib import Path
import argparse
import re
import pandas as pd
import fcntl
import json
from time import sleep

FILE_PATH = os.path.dirname(__file__)
sys.path.append(FILE_PATH)

algos = [
    "HumanoidNEAR", 
    # "HumanoidAMP"
]

motions = [
    # "amp_humanoid_walk.yaml",
    # "amp_humanoid_turning_walk.yaml",
    # "amp_humanoid_run.yaml",
    # "amp_humanoid_turning_run.yaml",
    # "amp_humanoid_crane_pose.yaml",
    # "amp_humanoid_single_left_punch.yaml",
    # "amp_humanoid_zombie_walk.yaml",
    # "amp_humanoid_bow.yaml",
    # "amp_humanoid_marching.yaml",
    # "amp_humanoid_tai_chi.yaml",
    # "amp_humanoid_mummy_walk.yaml",
    # "amp_humanoid_single_cartwheel.yaml",
    # "amp_humanoid_spin_kick.yaml",
    # "amp_humanoid_locate_and_strike.yaml",
    "amp_humanoid_walk_and_wave"
]

exp_type = ''

task_specific_cfg = {
    "amp_humanoid_walk.yaml": "headless=True max_iterations=60e6 num_envs=4096 ++train.params.config.minibatch_size=8192",
    "amp_humanoid_turning_walk.yaml": f"headless=True max_iterations=100e6 num_envs=4096 ++train.params.config.minibatch_size=8192 ++task.env.localRootObs=True ++task.env.envAssets=[\"{exp_type}\"]",
    "amp_humanoid_turning_run.yaml": f"headless=True max_iterations=100e6 num_envs=4096 ++train.params.config.minibatch_size=8192 ++task.env.localRootObs=True ++task.env.envAssets=[\"{exp_type}\"]",
    "amp_humanoid_run.yaml":"headless=True max_iterations=60e6 num_envs=4096 ++train.params.config.minibatch_size=8192",
    "amp_humanoid_crane_pose.yaml":"headless=True max_iterations=60e6 num_envs=4096 ++train.params.config.minibatch_size=8192",
    "amp_humanoid_single_left_punch.yaml":"headless=True max_iterations=80e6 num_envs=4096 ++train.params.config.minibatch_size=8192",
    "amp_humanoid_tai_chi.yaml":"headless=True max_iterations=100e6 num_envs=4096 ++train.params.config.minibatch_size=8192",
    "amp_humanoid_zombie_walk.yaml":"headless=True max_iterations=80e6 num_envs=4096 ++train.params.config.minibatch_size=8192",
    "amp_humanoid_bow.yaml":"headless=True max_iterations=80e6 num_envs=4096 ++train.params.config.minibatch_size=8192",
    "amp_humanoid_marching.yaml":"headless=True max_iterations=80e6 num_envs=4096 ++train.params.config.minibatch_size=8192",
    "amp_humanoid_mummy_walk.yaml":"headless=True max_iterations=80e6 num_envs=4096 ++train.params.config.minibatch_size=8192",
    "amp_humanoid_mummy_walk.yaml":f"headless=True max_iterations=100e6 num_envs=4096 ++train.params.config.minibatch_size=8192 ++task.env.localRootObs=True ++task.env.envAssets=[\"{exp_type}\"]",
    "amp_humanoid_single_cartwheel.yaml":"headless=True max_iterations=80e6 num_envs=4096 ++train.params.config.minibatch_size=8192 ",
    "amp_humanoid_spin_kick.yaml":"headless=True max_iterations=100e6 num_envs=4096 ++train.params.config.minibatch_size=8192 ++task.env.stateInit=WeightedRandom ++task.env.episodeLength=100",
    "amp_humanoid_locate_and_strike.yaml": f"headless=True max_iterations=100e6 num_envs=4096 ++train.params.config.minibatch_size=8192 ++task.env.localRootObs=True ++task.env.envAssets=[\"{exp_type}\"]",
    "amp_humanoid_walk_and_wave": "headless=True max_iterations=80e6 num_envs=4096 ++train.params.config.minibatch_size=8192 ++task.env.localRootObs=True",
}

near_task_specific_cfg = {
    "amp_humanoid_walk.yaml": "++train.params.config.near_config.training.n_iters=150000",
    "amp_humanoid_turning_walk.yaml": "++train.params.config.near_config.training.n_iters=150000",
    "amp_humanoid_run.yaml": "++train.params.config.near_config.training.n_iters=150000",
    "amp_humanoid_turning_run.yaml": "++train.params.config.near_config.training.n_iters=120000",
    "amp_humanoid_crane_pose.yaml": "++train.params.config.near_config.training.n_iters=100000",
    "amp_humanoid_single_left_punch.yaml": "++train.params.config.near_config.training.n_iters=120000",
    "amp_humanoid_tai_chi.yaml": "++train.params.config.near_config.training.n_iters=150000",
    "amp_humanoid_zombie_walk.yaml": "++train.params.config.near_config.training.n_iters=100000",
    "amp_humanoid_bow.yaml": "++train.params.config.near_config.training.n_iters=100000",
    "amp_humanoid_marching.yaml": "++train.params.config.near_config.training.n_iters=150000",
    "amp_humanoid_mummy_walk.yaml": "++train.params.config.near_config.training.n_iters=80000",
    "amp_humanoid_single_cartwheel.yaml": "++train.params.config.near_config.training.n_iters=80000",
    "amp_humanoid_spin_kick.yaml": "++train.params.config.near_config.training.n_iters=120000",
    "amp_humanoid_locate_and_strike.yaml": "++train.params.config.near_config.training.n_iters=120000",
    "amp_humanoid_walk_and_wave": {"amp_humanoid_walk.yaml": "++train.params.config.near_config.training.n_iters=150000 ++train.params.config.near_config.model.feature_mask=lower_body",
                                   "amp_humanoid_hello.yaml": "++train.params.config.near_config.training.n_iters=80000 ++train.params.config.near_config.model.feature_mask=upper_body", 
                                },
}

amp_task_specific_cfg = {
    "amp_humanoid_walk.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_turning_walk.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_run.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_turning_run.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_crane_pose.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_single_left_punch.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_tai_chi.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_zombie_walk.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_bow.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_marching.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_mummy_walk.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_single_cartwheel.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_spin_kick.yaml": "++train.params.config.amp_minibatch_size=4096",
    "amp_humanoid_locate_and_strike.yaml": "++train.params.config.amp_minibatch_size=4096",
}


manual_seeds = [42, 700, 8125, 97, 3538]

def generate_seeds(start=0, end=int(1e4), k=3, manual_seeds=[]):

    if manual_seeds != []:
        seeds = manual_seeds
    else:
        seeds = random.sample(range(start, end), k)

    return seeds

def generate_train_commands():
    train_cmds = Path(os.path.join(FILE_PATH, "train_cmds.pkl"))
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
                    if motion == "amp_humanoid_single_cartwheel.yaml":
                        base_cmd = [f"task={algo}Hands train={algo}PPO ++task.env.motion_file={motion} seed={seed} {task_specific_cfg[motion]}", f"{algo}_{os.path.splitext(motion)[0].replace('amp_humanoid_', '')}_{seed}"]
                    elif motion == "amp_humanoid_walk_and_wave":
                        base_cmd = [f"task={algo} seed={seed} {task_specific_cfg[motion]}", f"{algo}_{os.path.splitext(motion)[0].replace('amp_humanoid_', '')}_{seed}"]
                    else:
                        base_cmd = [f"task={algo} ++task.env.motion_file={motion} seed={seed} {task_specific_cfg[motion]}", f"{exp_type.upper()}_{algo}_{os.path.splitext(motion)[0].replace('amp_humanoid_', '')}_{seed}"]

                    if algo == "HumanoidNEAR":    
                        if motion == "amp_humanoid_walk_and_wave":
                            base_cmd[0] += " ++train.params.config.near_config.inference.randomise_init_motions=True ++train.params.config.near_config.inference.random_init_motion_ratio=1.0"

                            ncsn_cmds = []
                            eb_model_checkpoint = {}
                            running_mean_std_checkpoint = {}
                            for motion_file, cfg in near_task_specific_cfg[motion].items():
                                experiment_name = f"{base_cmd[1]}_{os.path.splitext(motion_file)[0].replace('amp_humanoid_', '')}_NCSN"
                                ncsn_cmd = base_cmd[0] + f" experiment={experiment_name}" + f" {cfg}" + f" ++task.env.motion_file={motion_file}"
                                ncsn_cmds.append(ncsn_cmd)
                                eb_model_checkpoint[motion_file] = f"ncsn_runs/{experiment_name}/nn/checkpoint.pth"
                                running_mean_std_checkpoint[motion_file] = f"ncsn_runs/{experiment_name}/nn/running_mean_std.pth"
                            
                            ncsn_dir = base_cmd[1]
                            w_task = 0.0
                            w_style = 1.0

                            eb_model_checkpoint = str(eb_model_checkpoint).replace('\'', '').replace(" ", "")
                            running_mean_std_checkpoint = str(running_mean_std_checkpoint).replace('\'', '').replace(" ", "") 

                            rl_cmd = base_cmd[0] + f" ++train.params.config.near_config.inference.eb_model_checkpoint={eb_model_checkpoint}" \
                            + f" ++train.params.config.near_config.inference.running_mean_std_checkpoint={running_mean_std_checkpoint}" + f" experiment={base_cmd[1]}" \
                            + f" ++train.params.config.near_config.inference.task_reward_w={w_task} ++train.params.config.near_config.inference.energy_reward_w={w_style}" \
                            + f" ++train.params.config.near_config.inference.composed_feature_mask=[lower_body,upper_body] ++task.env.motion_file=amp_humanoid_walk.yaml"

                            pending_cmds.append([ncsn_cmds, rl_cmd, False, False, False])
                            counter += 1

                        else:
                            if motion == "amp_humanoid_locate_and_strike.yaml":
                                base_cmd[0] += " ++train.params.config.near_config.inference.randomise_init_motions=True +train.params.config.near_config.inference.random_init_motion_files=[amp_humanoid_walk.yaml,amp_humanoid_single_left_punch.yaml]"

                            ncsn_cmd = base_cmd[0] + f" experiment={base_cmd[1]}" + f" {near_task_specific_cfg[motion]}"
                            
                            ncsn_dir = base_cmd[1]
                            w_task = 0.5
                            w_style = 0.5
                            eb_model_checkpoint = f"ncsn_runs/{ncsn_dir}/nn/checkpoint.pth"
                            running_mean_std_checkpoint = f"ncsn_runs/{ncsn_dir}/nn/running_mean_std.pth"
                            rl_cmd = base_cmd[0] + f" ++train.params.config.near_config.inference.eb_model_checkpoint={eb_model_checkpoint}" \
                            + f" ++train.params.config.near_config.inference.running_mean_std_checkpoint={running_mean_std_checkpoint}" + f" experiment={base_cmd[1]}" \
                            + f" ++train.params.config.near_config.inference.task_reward_w={w_task} ++train.params.config.near_config.inference.energy_reward_w={w_style}"

                            pending_cmds.append([ncsn_cmd, rl_cmd, False, False, False])
                            counter += 1
                    elif algo == "HumanoidAMP":
                        ncsn_cmd = ""
                        task_reward_w = 0.5
                        disc_reward_w = 0.5

                        if motion == "amp_humanoid_locate_and_strike.yaml":
                            base_cmd[0] += " ++train.params.config.randomise_init_motions=True +train.params.config.random_init_motion_files=[amp_humanoid_walk.yaml,amp_humanoid_single_left_punch.yaml]"

                        rl_cmd = base_cmd[0] + f" experiment={base_cmd[1]}" + f" {amp_task_specific_cfg[motion]}" \
                        + f" ++train.params.config.task_reward_w={task_reward_w} ++train.params.config.disc_reward_w={disc_reward_w}"

                        pending_cmds.append([ncsn_cmd, rl_cmd, False, False, False])
                        counter += 1


        
        cmds = [{"algos": algos, "motions":motions, "seeds":seeds, "pending_cmds":pending_cmds, "num_runs": counter}]
        with open(os.path.join(FILE_PATH, "train_cmds.yaml"), 'w') as yaml_file:
            yaml.dump(cmds, yaml_file, default_flow_style=False)

        pending_cmds_df = pd.DataFrame(pending_cmds, columns=["ncsn_cmd", "rl_cmd", "job_assigned", "ncsn_cmd_passed", "rl_cmd_passed"])
        pending_cmds_df.to_pickle(os.path.join(FILE_PATH, "train_cmds.pkl"))

        # print(f"Train commands generated! Please manually delete the file {train_cmds} to create a new one and call this script again to start training")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="type of model to train (reinforcement learning or NCSN)")
    parser.add_argument("-jid", "--job_idx", type=int, required=True, help="ID of the command assigned to a job") 
    parser.add_argument("-k", "--kth_ncsn", type=int, required=False, help="In case of composed energy function, which one to train")    
    args = parser.parse_args()

    # Does nothing if cmds already exist
    # generate_train_commands()

    for _ in range(3):
        try:
            with open(os.path.join(FILE_PATH, "train_cmds.pkl"), "a") as file:
                # Acquire exclusive lock on the file
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)

                # Perform operations on the file
                # open the training commands file
                cmds = pd.read_pickle(os.path.join(FILE_PATH, "train_cmds.pkl"))

                if args.model == "ncsn":
                    # non_assigned = cmds[cmds['job_assigned']==False]
                    # if non_assigned.empty:
                    #     print("done")
                    #     quit()

                    if args.job_idx == None:
                        print("done")
                        quit()

                    else:
                        # first_non_assigned = non_assigned.loc[non_assigned.index.min()]
                        # job_idx = first_non_assigned.name
                        job_idx = args.job_idx
                        job_cmds = cmds.loc[job_idx]
                        command_to_pass = job_cmds["ncsn_cmd"]
                        if isinstance(command_to_pass, list):
                            command_to_pass = command_to_pass[args.kth_ncsn]
                        # cmds.loc[job_idx, "job_assigned"] = True
                        # cmds.loc[job_idx, "ncsn_cmd_passed"] = True
                        # cmds.to_pickle(os.path.join(FILE_PATH, "train_cmds.pkl"))
                        # output = {"cmd":command_to_pass, "job_idx":int(job_idx)}
                        # print(json.dumps(output))
                        print(command_to_pass)
                        

                elif args.model == "rl":
                    # non_assigned = cmds[(cmds['job_assigned']==True) & (cmds['rl_cmd_passed']==False)]
                    # if non_assigned.empty:
                    #     print("done")
                    #     quit()

                    if args.job_idx == None:
                        print("done")
                        quit()

                    else:
                        job_idx = args.job_idx
                        job_cmds = cmds.loc[job_idx]
                        command_to_pass = job_cmds["rl_cmd"]
                        # cmds.loc[job_idx, "rl_cmd_passed"] = True
                        # cmds.to_pickle(os.path.join(FILE_PATH, "train_cmds.pkl"))
                        print(command_to_pass)

                # Release the lock
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                break

        except Exception as e:
            print(e)
            sleep(1.0)

    



 


