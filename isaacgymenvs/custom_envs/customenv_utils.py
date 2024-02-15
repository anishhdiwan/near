from rl_games.common.tr_helpers import dicts_to_dict_with_arrays
from rl_games.common.ivecenv import IVecEnv
from rl_games.common.algo_observer import AlgoObserver
import numpy as np
import gym
import random
from time import sleep
import torch

from .motion_lib import MotionLib

class PushTAlgoObserver(AlgoObserver):
    ## TODO: Figure out a way to record episode returns with asynchronous environments...
    def __init__(self):
        super().__init__()
        self.algo = None
        self.writer = None

        self.current_epoch = 0
        self.episode_cumulative_return = None


    def after_init(self, algo):
        self.algo = algo
        self.writer = self.algo.writer


    def process_infos(self, infos, done_indices):
        # print(f"Current Epoch {self.current_epoch} | Done idx {done_indices}")
        if not infos:
            return

        # done_indices = done_indices.cpu().numpy()

        if isinstance(infos, dict):
            if 'scores' in infos:
                # Set up the cumulative return array at first
                if self.episode_cumulative_return is None:
                    self.episode_cumulative_return = np.zeros(infos['scores'].shape)
                
                for idx in range(len(infos['scores'])):
                    # Go through all idices. If that env is done then don't update its episode return
                    if idx not in done_indices:
                        self.episode_cumulative_return[idx] += infos['scores'][idx]


    def after_clear_stats(self):
        self.episode_cumulative_return = None

    def after_print_stats(self, frame, epoch_num, total_time):
        
        # Log episode returns only for each new epoch
        if epoch_num > self.current_epoch:
            # Update the current epoch 
            self.current_epoch = epoch_num   
            if self.writer is not None:
                mean_ep_scores = self.episode_cumulative_return.mean()
                # self.writer.add_scalar('scores/mean', mean_scores, frame)
                self.writer.add_scalar('scores/ep_returns', mean_ep_scores, epoch_num)
                # self.writer.add_scalar('scores/time', mean_scores, total_time)

            self.episode_cumulative_return = None




class CustomRayWorker:
    def __init__(self, config_dict, config_name, config):
        self.env = config_dict[config_name]['env_creator'](**config)

    def _obs_to_fp32(self, obs):
        if isinstance(obs, dict):
            for k, v in obs.items():
                if isinstance(v, dict):
                    for dk, dv in v.items():
                        if dv.dtype == np.float64:
                            v[dk] = dv.astype(np.float32)
                else:
                    if v.dtype == np.float64:
                        obs[k] = v.astype(np.float32)
        else:
            if obs.dtype == np.float64:
                obs = obs.astype(np.float32)
        return obs

    def step(self, action):
        next_state, reward, is_done, info = self.env.step(action)
        
        if np.isscalar(is_done):
            episode_done = is_done
        else:
            episode_done = is_done.all()
        if episode_done:
            next_state = self.reset()
        next_state = self._obs_to_fp32(next_state)
        return next_state, reward, is_done, info

    def seed(self, seed):
        if hasattr(self.env, 'seed'):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.env.seed(seed)
            
    def render(self):
        self.env.render()

    def reset(self):
        obs = self.env.reset()
        obs = self._obs_to_fp32(obs)
        return obs

    def get_action_mask(self):
        return self.env.get_action_mask()

    def get_number_of_agents(self):
        if hasattr(self.env, 'get_number_of_agents'):
            return self.env.get_number_of_agents()
        else:
            return 1

    def set_weights(self, weights):
        self.env.update_weights(weights)

    def can_concat_infos(self):
        if hasattr(self.env, 'concat_infos'):
            return self.env.concat_infos
        else:
            return False

    def get_env_info(self):
        info = {}
        observation_space = self.env.observation_space

        # Added to provide shape info to amp_continuous
        if hasattr(self.env, "amp_observation_space"):
            info['amp_observation_space'] = self.env.amp_observation_space

        #if isinstance(observation_space, gym.spaces.dict.Dict):
        #    observation_space = observation_space['observations']

        info['action_space'] = self.env.action_space
        info['observation_space'] = observation_space
        info['state_space'] = None
        info['use_global_observations'] = False
        info['agents'] = self.get_number_of_agents()
        info['value_size'] = 1
        if hasattr(self.env, 'use_central_value'):
            info['use_global_observations'] = self.env.use_central_value
        if hasattr(self.env, 'value_size'):
            info['value_size'] = self.env.value_size
        if hasattr(self.env, 'state_space'):
            info['state_space'] = self.env.state_space
        return info
    
    def get_num_amp_obs_per_step(self):
        return self.env._num_amp_obs_per_step

    def get_num_amp_obs_steps(self):
        return self.env._num_amp_obs_steps

    def get_motion_file(self):
        return self.env._motion_file


class CustomRayVecEnv(IVecEnv):
    import ray

    def __init__(self, config_dict, config_name, num_actors, **kwargs):
        # Explicityly passing in the dictionary containing env_name: {vecenv_type, env_creator}
        self.config_dict = config_dict
        # Pointing the env variable to the object instance. This enables running the _fetch_amp_obs_demo function without any modifications to amp_continuous
        self.env = self

        self.config_name = config_name
        self.num_actors = num_actors
        self.use_torch = False
        self.seed = kwargs.pop('seed', None)

        
        self.remote_worker = self.ray.remote(CustomRayWorker)
        self.workers = [self.remote_worker.remote(self.config_dict, self.config_name, kwargs) for i in range(self.num_actors)]

        if self.seed is not None:
            seeds = range(self.seed, self.seed + self.num_actors)
            seed_set = []
            for (seed, worker) in zip(seeds, self.workers):	        
                seed_set.append(worker.seed.remote(seed))
            self.ray.get(seed_set)

        res = self.workers[0].get_number_of_agents.remote()
        self.num_agents = self.ray.get(res)

        res = self.workers[0].get_env_info.remote()
        env_info = self.ray.get(res)
        res = self.workers[0].can_concat_infos.remote()
        can_concat_infos = self.ray.get(res)        
        self.use_global_obs = env_info['use_global_observations']
        self.concat_infos = can_concat_infos
        self.obs_type_dict = type(env_info.get('observation_space')) is gym.spaces.Dict
        self.state_type_dict = type(env_info.get('state_space')) is gym.spaces.Dict
        if self.num_agents == 1:
            self.concat_func = np.stack
        else:
            self.concat_func = np.concatenate

        # Set up the motions library
        self._setup_motions()
    
    def step(self, actions):
        newobs, newstates, newrewards, newdones, newinfos = [], [], [], [], []
        res_obs = []
        if self.num_agents == 1:
            for (action, worker) in zip(actions, self.workers):	        
                res_obs.append(worker.step.remote(action))
        else:
            for num, worker in enumerate(self.workers):
                res_obs.append(worker.step.remote(actions[self.num_agents * num: self.num_agents * num + self.num_agents]))

        all_res = self.ray.get(res_obs)
        for res in all_res:
            cobs, crewards, cdones, cinfos = res
            if self.use_global_obs:
                newobs.append(cobs["obs"])
                newstates.append(cobs["state"])
            else:
                newobs.append(cobs)
            newrewards.append(crewards)
            newdones.append(cdones)
            newinfos.append(cinfos)

        if self.obs_type_dict:
            ret_obs = dicts_to_dict_with_arrays(newobs, self.num_agents == 1)
        else:
            ret_obs = self.concat_func(newobs)

        if self.use_global_obs:
            newobsdict = {}
            newobsdict["obs"] = ret_obs
            
            if self.state_type_dict:
                newobsdict["states"] = dicts_to_dict_with_arrays(newstates, True)
            else:
                newobsdict["states"] = np.stack(newstates)            
            ret_obs = newobsdict
        # if self.concat_infos:
        newinfos = dicts_to_dict_with_arrays(newinfos, False)
        
        # print(newinfos)
        return ret_obs, self.concat_func(newrewards), self.concat_func(newdones), newinfos

    def get_env_info(self):
        res = self.workers[0].get_env_info.remote()
        return self.ray.get(res)

    def set_weights(self, indices, weights):
        res = []
        for ind in indices:
            res.append(self.workers[ind].set_weights.remote(weights))
        self.ray.get(res)

    def has_action_masks(self):
        return True

    def get_action_masks(self):
        mask = [worker.get_action_mask.remote() for worker in self.workers]
        masks = self.ray.get(mask)
        return np.concatenate(masks, axis=0)

    def reset(self):
        res_obs = [worker.reset.remote() for worker in self.workers]
        newobs, newstates = [],[]
        for res in res_obs:
            cobs = self.ray.get(res)
            if self.use_global_obs:
                newobs.append(cobs["obs"])
                newstates.append(cobs["state"])
            else:
                newobs.append(cobs)

        if self.obs_type_dict:
            ret_obs = dicts_to_dict_with_arrays(newobs, self.num_agents == 1)
        else:
            ret_obs = self.concat_func(newobs)

        if self.use_global_obs:
            newobsdict = {}
            newobsdict["obs"] = ret_obs
            
            if self.state_type_dict:
                newobsdict["states"] = dicts_to_dict_with_arrays(newstates, True)
            else:
                newobsdict["states"] = np.stack(newstates)            
            ret_obs = newobsdict
        return ret_obs

    def reset_done(self):
        """
        Added as a wrapper around reset() to enable compatibility with the amp_continuous algo
        """
        return self.reset()

    def fetch_amp_obs_demo(self, num_samples):
        """
        Fetch a set of num_samples demo observations from the environment motionlib
        """

        amp_obs_demo = self._motion_lib.sample_motions(num_samples)


    def _setup_motions(self):
        """
        Set up the motion library
        """

        # Pulling config data from the env
        res = self.workers[0].get_num_amp_obs_steps.remote()
        num_amp_obs_steps = self.ray.get(res)

        res = self.workers[0].get_num_amp_obs_per_step.remote()
        num_amp_obs_per_step = self.ray.get(res)

        res = self.workers[0].get_motion_file.remote()
        motion_file = self.ray.get(res)
        
        # TODO: set device=self.device
        self._motion_lib = MotionLib(motion_file, num_amp_obs_steps, num_amp_obs_per_step)
