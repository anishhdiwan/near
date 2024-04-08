

# Importing from the file path
import sys
import os
FILE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(FILE_PATH)
from pymunk_override import DrawOptions

from motion_lib import MotionLib, unnormalize_data
from pusht_env import PushTEnv
from maze_env import MazeEnv
import time
from tqdm import tqdm
# import matplotlib.pyplot as plt

environment = "maze" # "pushT"

if environment == "maze":
    env = MazeEnv() 
    motion_file = FILE_PATH + "/data/maze_env/maze_motions.zarr"
    num_obs_steps = 2
    num_obs_per_step = 2
    auto_ends = False

elif environment == "pushT":
    env = PushTEnv() 
    motion_file = FILE_PATH + "/data/pusht/pusht_cchi_v7_replay.zarr"
    num_obs_steps = 2
    num_obs_per_step = 5
    auto_ends = True


num_samples = 512
mode = "human"
num_plays = 10
sleep_time = 0.001



def test_non_episodic_dataloader():
    # episodic=True (default) returns trajectories of length num_samples, where each trajectory is guaranteed to be from a single episode (no trajectories with episode ends)
    # episodic=False just samples in an unshuffled manner from the whole sequence of trajectories
    motion_lib = MotionLib(motion_file, num_obs_steps, num_obs_per_step, episodic=False, auto_ends=auto_ends, test_split=False)

    # A dataloader can also be returned to get shuffled or unshuffled samples of some required length. When shuffle=True, the dataloader essentially returns trajectories regardless of episode ends
    dataloader = motion_lib.get_traj_agnostic_dataloader(batch_size=num_samples, shuffle=False)

    print("Playing unshuffled, trajectory agnostic motion data")
    obs = env.reset()
    for i, X in enumerate(dataloader):
        print(f"Batch {i}, {X.shape}")

        if environment == "pushT":
            unpaired_obs = X[:, :5]
        elif environment == "maze":
            unpaired_obs = X[:, :2]

        for idx, obs in enumerate(tqdm(unpaired_obs)):
            n_steps = env.sim_hz // env.control_hz
            dt = 1.0 / env.sim_hz

            for i in range(n_steps):
                env.space.step(dt)

            env.agent.position = tuple(obs[:2]) 
            try:
                env.block.position = tuple(obs[2:4])
                env.block.angle = obs[4]
            except Exception:
                pass
            env.render(mode=mode)
            time.sleep(sleep_time)


def play_sampled_trajectories(num_plays=num_plays, num_samples=num_samples):
    motion_lib = MotionLib(motion_file, num_obs_steps, num_obs_per_step, auto_ends=auto_ends, test_split=False)
    for _ in range(num_plays):
        obs = env.reset()
        print("ENV RESET")
        paired_obs_demo = motion_lib.sample_motions(num_samples)

        if environment == "pushT":
            unpaired_obs = paired_obs_demo[:, :5]
        elif environment == "maze":
            unpaired_obs = paired_obs_demo[:, :2]

        # unpaired_obs = unnormalize_data(unpaired_obs, motion_lib.dataset.stats['obs'])
        print(f"Sampled Obs: {paired_obs_demo.shape}")
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
            try:
                env.block.position = tuple(obs[2:4].numpy())
                env.block.angle = obs[4].item()
            except Exception:
                pass
            env.render(mode=mode)
            time.sleep(sleep_time)


def play_all_episodes():
    motion_lib = MotionLib(motion_file, num_obs_steps, num_obs_per_step, auto_ends=auto_ends, test_split=False)
    # VIEW ALL EPISODES
    paired_processed_episodes = motion_lib.get_episodes()
    print(f"VIEWING ALL {len(paired_processed_episodes)} EPISODES")


    for i, episode in enumerate(paired_processed_episodes):
        obs = env.reset()
        print(f"EPISODE {i} | ENV RESET")
        # episode = paired_processed_episodes[5]
        if environment == "pushT":
            unpaired_obs = episode[:, :5]
        elif environment == "maze":
            unpaired_obs = episode[:, :2]
        # unpaired_obs = unnormalize_data(unpaired_obs, motion_lib.dataset.stats['obs'])
        # print(f"Episode Unpaired Obs (len episode): {unpaired_obs.shape}")

        for idx, obs in enumerate(tqdm(unpaired_obs)):
            # print(f"step {idx} observation {obs}")
            # action = env.action_space.sample()
            # _, _, _, _ = env.step(action)

            n_steps = env.sim_hz // env.control_hz
            dt = 1.0 / env.sim_hz

            for i in range(n_steps):
                env.space.step(dt)

            env.agent.position = tuple(obs[:2]) 
            try:
                env.block.position = tuple(obs[2:4])
                env.block.angle = obs[4]
            except Exception:
                pass
            env.render(mode=mode)
            time.sleep(sleep_time)



# Calling functions
# play_sampled_trajectories()
# play_all_episodes()
test_non_episodic_dataloader()