from maze_env import MazeEnv
import time

env = MazeEnv(normalise_action=False)
env.teleop_agent(record_data=True)
