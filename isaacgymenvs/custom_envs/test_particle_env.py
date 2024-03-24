
from particle_env import ParticleEnv
import time

env = ParticleEnv()

obs = env.reset()


for _ in range(200):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(f"Obs {observation} | Rew {reward} | done {done} | info {info}")
    env.render(mode="human")
    time.sleep(0.05)