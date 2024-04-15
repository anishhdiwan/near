from maze_env import MazeEnv
import time
import numpy as np

# env = MazeEnv()
# env.teleop_agent(record_data=True)


# env = MazeEnv(normalise_action=True)
# obs = env.reset()

# for _ in range(100):
#     action = np.array([1., 1.]) # env.action_space.sample()
#     print(np.array(tuple(env.agent.position)))
#     observation, reward, done, info = env.step(action)
#     print(reward)
#     env.render(mode="human")
#     time.sleep(0.1)

# env.close()



## Visualising Saved Motions ##
# _01-18-50-00.npy
# _01-18-50-14.npy
# _01-18-50-43.npy

# states = np.load('data/maze_env/_10-22-08-24.npy')
# print(len(states))

# env = MazeEnv()
# obs = env.reset()

# dt = 1.0 / env.sim_hz
# n_steps = env.sim_hz // env.control_hz


# for pos in states:

#     for i in range(n_steps):
#         env.agent.position = pos.tolist()
#         # Step physics.
#         env.space.step(dt)
        
#     env.render(mode="human")