import time

from particle_env import ParticleEnv

env = ParticleEnv()

obs = env.reset()

for _ in range(200):
    action = env.action_space.sample()
    _, _, _, _ = env.step(action)
    env.render(mode="human")
    time.sleep(0.5)