


from custom_envs.motion_lib import MotionLib, unnormalize_data
from custom_envs.pusht_env import PushTEnv
import time
# import matplotlib.pyplot as plt

env = PushTEnv() # cfg is obtained from the config file. This is passed in within the algo init step as a kwarg
motion_file = "custom_envs/data/pusht/pusht_cchi_v7_replay.zarr"
num_amp_obs_steps = 2
num_amp_obs_per_step = 5
num_samples = 128
mode = "human"
num_plays = 8

motion_lib = MotionLib(motion_file, num_amp_obs_steps, num_amp_obs_per_step, episodic=False)
# print(motion_lib.dataset.stats)



# dataloader = motion_lib.get_traj_agnostic_dataloader(batch_size=num_samples, shuffle=True)
# for idx, X in enumerate(dataloader):
#     print(f"{idx}, {X.shape}")
# quit()

for _ in range(num_plays):
    obs = env.reset()
    print("ENV RESET")
    amp_obs_demo = motion_lib.sample_motions(num_samples)
    unpaired_obs = amp_obs_demo[:, :5]
    # unpaired_obs = unnormalize_data(unpaired_obs, motion_lib.dataset.stats['obs'])
    print(f"Sampled Obs: {amp_obs_demo.shape}")
    print(f"Sampled Unpaired Obs: {unpaired_obs.shape}")

    for idx, obs in enumerate(unpaired_obs):
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

