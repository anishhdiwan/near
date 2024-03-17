

# Importing from the file path
import sys
import os
FILE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(FILE_PATH)
from pymunk_override import DrawOptions

from motion_lib import MotionLib, unnormalize_data
from pusht_env import PushTEnv
import time
from tqdm import tqdm
# import matplotlib.pyplot as plt

env = PushTEnv() 
motion_file = FILE_PATH + "/data/pusht/pusht_cchi_v7_replay.zarr"
num_amp_obs_steps = 2
num_amp_obs_per_step = 5
num_samples = 128
mode = "human"
num_plays = 10


# episodic=True (default) returns trajectories of length num_samples, where each trajectory is guaranteed to be from a single episode (no trajectories with episode ends)
# episodic=False just samples in an unshuffled manner from the whole sequence of trajectories
motion_lib = MotionLib(motion_file, num_amp_obs_steps, num_amp_obs_per_step)
# print(motion_lib.dataset.stats)

# A dataloader can also be returned to get shuffled or unshuffled samples of some required length. When shuffle=True, the dataloader essentially returns trajectories regardless of episode ends
# dataloader = motion_lib.get_traj_agnostic_dataloader(batch_size=num_samples, shuffle=True)
# for idx, X in enumerate(dataloader):
#     print(f"{idx}, {X.shape}")


def play_sampled_trajectories(num_plays=num_plays, num_samples=num_samples):
    for _ in range(num_plays):
        obs = env.reset()
        print("ENV RESET")
        amp_obs_demo = motion_lib.sample_motions(num_samples)
        unpaired_obs = amp_obs_demo[:, :5]
        # unpaired_obs = unnormalize_data(unpaired_obs, motion_lib.dataset.stats['obs'])
        print(f"Sampled Obs: {amp_obs_demo.shape}")
        print(f"Sampled Unpaired Obs: {unpaired_obs.shape}")

        for idx, obs in enumerate(tqdm(unpaired_obs)):
            # print(f"step {idx} observation {obs}")
            # action = env.action_space.sample()
            # _, _, _, _ = env.step(action)

            n_steps = env.sim_hz // env.control_hz
            dt = 1.0 / env.sim_hz

            for i in range(n_steps):
                env.space.step(dt)

            env.agent.position = tuple(obs[:2].numpy()) 
            env.block.position = tuple(obs[2:4].numpy())
            env.block.angle = obs[4].item()
            env.render(mode=mode)
            time.sleep(0.1)


def play_all_episodes():
    # VIEW ALL EPISODES
    print(f"VIEWING ALL {len(paired_processed_episodes)} EPISODES")
    paired_processed_episodes = motion_lib.get_episodes()

    for episode in paired_processed_episodes:
        obs = env.reset()
        print("EPISODE END! ENV RESET")
        unpaired_obs = episode[:, :5]
        # unpaired_obs = unnormalize_data(unpaired_obs, motion_lib.dataset.stats['obs'])
        print(f"Episode Unpaired Obs (len episode): {unpaired_obs.shape}")

        for idx, obs in enumerate(tqdm(unpaired_obs)):
            # print(f"step {idx} observation {obs}")
            # action = env.action_space.sample()
            # _, _, _, _ = env.step(action)

            n_steps = env.sim_hz // env.control_hz
            dt = 1.0 / env.sim_hz

            for i in range(n_steps):
                env.space.step(dt)

            env.agent.position = tuple(obs[:2]) 
            env.block.position = tuple(obs[2:4])
            env.block.angle = obs[4]
            env.render(mode=mode)
            time.sleep(0.1)


# Calling functions
play_sampled_trajectories()
# play_all_episodes()