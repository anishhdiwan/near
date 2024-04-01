from maze_env import MazeEnv
import time
import numpy as np

env = MazeEnv()
env.teleop_agent(record_data=True)


# env = MazeEnv(normalise_action=True)
# obs = env.reset()

# for _ in range(env.max_env_steps):
#     action = np.array([0., 1.]) # env.action_space.sample()
#     observation, reward, done, info = env.step(action)
#     env.render(mode="human")
#     # time.sleep(0.1)

# env.close()


