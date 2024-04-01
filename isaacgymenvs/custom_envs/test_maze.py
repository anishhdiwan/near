from maze_env import MazeEnv
import time

env = MazeEnv()
env.teleop_agent(record_data=True)


# env = MazeEnv(normalise_action=False)
# obs = env.reset()

# for _ in range(200):
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
#     env.render(mode="human")
#     time.sleep(0.1)

# env.close()


