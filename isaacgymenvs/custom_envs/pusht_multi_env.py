from pusht_single_env import PushTEnv
import concurrent.futures
import random
import cv2
import numpy as np

class MultiPushTEnv():
    """A, asynchronous parallelised version of the pushT environment.

    In asynchronous execution environments are stepped at the same time but are executed in parallel. This allows for environments to be reset independently

    Args:
        n_envs (int): number of environments to run
        max_workers (int): number of multithreading workers
    """
    def __init__(self, n_envs, asynchronous=False, max_workers=2):
        self.asynchronous = asynchronous
        self.n_envs = n_envs
        self.max_workers = max_workers
        
        # Init envs (seed 0-200 are used for the demonstration dataset)
        seeds = random.sample(range(201, 10000), self.n_envs)
        self.envs = []
        for i in range(n_envs):
            env = PushTEnv()
            # env.seed(seeds[i])
            self.envs.append(env)
        
    def step_single_env(self, env, action):
        """Step one environment with the action. Convenience method for threading pooling
        """
        obs, reward, done, info = env.step(action)
        return obs, reward, done, info


    def reset_single_env(self, env):
        """Reset one environment. Convenience method for threading pooling
        """
        return env.reset()

    def render_single_env(self, env):
        """Rended a single environment. Convenience method for threading pooling 
        """
        return env.render(mode='rgb_array')
    

    def step(self, actions):
        """Step all environments in parallel using threading. Return a list of their obs, reward, done, info. 
        If an env is done, then reset it automatically and return the new observations after resetting.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            step_outcomes = list(executor.map(self.step_single_env, self.envs, actions))

        # Reset an env if it is done
        reset_list = []
        for idx, step_outcome in enumerate(step_outcomes):
            if step_outcome[2] == True:
                reset_list.append(idx)

        if reset_list != []:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                new_obs = list(executor.map(self.reset_single_env, [self.envs[idx] for idx in reset_list]))
                
            return step_outcomes, {reset_list[i]: new_obs[i] for i in range(len(reset_list))}

        else:
            return step_outcomes, None


    def reset(self):
        """Reset all environments together in parallel. Done only at the start of the cycle
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self.reset_single_env, self.envs))


    def render(self):
        """
        Render all environments together in parallel
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self.render_single_env, self.envs))




### TESTING ###

multienv = MultiPushTEnv(n_envs=5)
obs = multienv.reset()

print(f"Observations on resetting \n {obs}")


# Sampling random actions
actions = []
for i in range(multienv.n_envs):
    actions.append(multienv.envs[0].action_space.sample())


# Stepping in parallel
step_outcomes, reset_obs = multienv.step(actions)

print(f"Step outcomes of each env (excluding info) \n {step_outcomes[0][:-1]}")

# Rendering
imgs = multienv.render()
cv2.imshow("frame", imgs[2])
# waits for user to press any key 
# (this is necessary to avoid Python kernel form crashing) 
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows() 


