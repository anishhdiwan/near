from rl_games.common.tr_helpers import dicts_to_dict_with_arrays
from rl_games.common.ivecenv import IVecEnv
from rl_games.common.algo_observer import AlgoObserver
import numpy as np
import gym
import random
from time import sleep
import torch
import time
import copy

from .gym_motion_lib import GymMotionLib

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
            self.env.seed(seed)
            print(f"Env instantiated with RNG seed {self.env._seed}")
            

    # Each ray worker operates with its own RNG with its own state
    def test_seed(self):
        print(self.env.np_random.__getstate__())
            
    def render(self):
        self.env.render(mode = 'human') 

    def reset(self, state=None):
        if state is None:
            obs = self.env.reset()
        else:
            obs = self.env.reset_to_state(state)
        
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

        # Added to provide shape info to amp_continuous and similar algorithms
        if hasattr(self.env, "observation_space"):
            # amp_observation_space and paired_observation_space contain the same information. 
            # amp_observation_space is added to maintain compatibility with adversatial motion priors
            info['amp_observation_space'] = self.env.paired_observation_space
            info['paired_observation_space'] = self.env.paired_observation_space

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
    
    def get_num_obs_per_step(self):
        return self.env._num_obs_per_step

    def get_num_obs_steps(self):
        return self.env._num_obs_steps

    def get_motion_file(self):
        return self.env._motion_file

    def get_num_envs(self):
        return self.env._num_envs
    
    def get_render_mode(self):
        return self.env._headless

    def get_training_algo(self):
        return self.env._training_algo


class CustomRayVecEnv(IVecEnv):
    import ray

    def __init__(self, config_dict, config_name, num_actors, **kwargs):

        # TODO: set one device globally
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set up a dummy variable for viewer to log debug info in amp_continuous
        self.viewer = True


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

        # Seed is passed via kwargs while the class is instantiated (done in the algo class in rl_games)
        print(f"Main seed {self.seed}. Setting different seeds for each parallel env")
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


        # Pulling config data from the env
        res = self.workers[0].get_training_algo.remote()
        self.training_algo = self.ray.get(res)

        # Set up motion library to sample motions if AMP
        if self.training_algo in ["AMP"]:
            # Set up the motions library
            self._setup_motions(setup_motionlib=True)
        # Only set up experience buffers for paired observations for DMP (saves unnecessary memory usage)
        elif self.training_algo in ["DMP", "CEM"]:
            # Set up the motions library
            self._setup_motions(setup_motionlib=False)

        # cfg for visualisation
        res = self.workers[0].get_render_mode.remote()
        self.headless = self.ray.get(res)
        self.done_envs = None

    def step_selected(self, indices, actions):
        """Step some selected environments

        Args:
            indices (list): list of env indices to step
            actions (np array): array of actions to take in each env
        """
        assert self.num_agents == 1, "step_selected() can only be applied for single-agent environments"
        newobs, newstates, newrewards, newdones, newinfos = [], [], [], [], []
        res_obs = []
        if self.num_agents == 1:
            for idx, worker in enumerate(self.workers):
                if idx in indices:
                    res_obs.append(worker.step.remote(actions[indices.index(idx)]))

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

        if self.training_algo in ["AMP", "DMP", "CEM"]:
            # Augmenting the infos dict
            self.post_step_procedures(ret_obs, indices=indices)
            newinfos = self.augment_infos(newinfos, newdones, indices=indices)

        # Render the environment (doesn't do anything if headless is True). Only render if one environment is stepped
        if len(indices) == 1:
            self.render()

        # print(newinfos)
        self.done_envs = np.nonzero(newdones)[0]
        self.last_obs = ret_obs

        return ret_obs, self.concat_func(newrewards), self.concat_func(newdones), newinfos

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

        if self.training_algo in ["AMP", "DMP", "CEM"]:
            # Augmenting the infos dict
            self.post_step_procedures(ret_obs)
            newinfos = self.augment_infos(newinfos, newdones)

        # Render the environment (doesn't do anything if headless is True)
        self.render()

        # print(newinfos)
        self.done_envs = np.nonzero(newdones)[0]
        self.last_obs = ret_obs

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

    def reset_selected(self, indices=[0], state=None):
        """
        Reset only the selected workers (environments). Optionally reset the workers to a specific state
        """

        res_obs = []
        for idx, worker in enumerate(self.workers):
            if idx in indices:
                res_obs.append(worker.reset.remote(state=state))

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
        
        self._past_obs_buf[indices] = torch.from_numpy(ret_obs).to(self.device)
        return ret_obs

    def reset_done(self):
        """
        Reset all done envs. If none are done, return the last seens observations as the reset observations. In the first step, reset all envs normally
        """
        if self.done_envs is None:
            # Return the reset observations with an empty dummy dict to match the return type expected by the amp_continuous algo
            obs = self.reset()
            # Set up the history amp obs
            self._past_obs_buf[:] = torch.from_numpy(obs)
            
            return obs, []
        
        else:

            # If no environments are done
            if len(self.done_envs) == 0:
                # No need to explicitly define past_obs_buf as no environments were done. It is directly set in post_step_procedures
                return copy.deepcopy(self.last_obs), []

            # If all envs are done
            elif len(self.done_envs) == len(self.workers):
                # Return the reset observations with an empty dummy dict to match the return type expected by the amp_continuous algo
                obs = self.reset()

                # Set up the history amp obs
                self._past_obs_buf[:] = torch.from_numpy(obs)
                return obs, []

            # If some are done and some are not
            else:
                # res_obs = [worker.reset.remote() for worker in self.workers]
                # Add all results of the reset function of the done envs
                res_obs = []
                for idx, worker in enumerate(self.workers):
                    if idx in self.done_envs:
                        res_obs.append(worker.reset.remote())
                
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

                last_obs = copy.deepcopy(self.last_obs)
                for idx, done_env_idx in enumerate(self.done_envs):
                    last_obs[done_env_idx] = ret_obs[idx]

                ret_obs = last_obs
                self._past_obs_buf[:] = torch.from_numpy(ret_obs)
                return ret_obs, self.done_envs


    def fetch_amp_obs_demo(self, num_samples):
        """
        Fetch a set of num_samples demo observations from the environment motionlib
        """

        amp_obs_demo = self._motion_lib.sample_motions(num_samples)
        return amp_obs_demo


    def _setup_motions(self, setup_motionlib=True):
        """
        Set up the motion library
        """

        # Pulling config data from the env
        res = self.workers[0].get_num_obs_steps.remote()
        num_obs_steps = self.ray.get(res)

        res = self.workers[0].get_num_obs_per_step.remote()
        num_obs_per_step = self.ray.get(res)

        self.num_obs = int(num_obs_steps * num_obs_per_step)

        res = self.workers[0].get_motion_file.remote()
        motion_file = self.ray.get(res)

        res = self.workers[0].get_num_envs.remote()
        num_envs = self.ray.get(res)
        
        if setup_motionlib:
            # TODO: set device=self.device
            self._motion_lib = GymMotionLib(motion_file, num_obs_steps, num_obs_per_step)

        # TODO: set device=self.device
        # Set up the observations buffer. This contains s-s' pairs and has the shape [num envs, num_obs_steps*num_obs_per_step]
        self._obs_buf = torch.zeros((num_envs, num_obs_steps, num_obs_per_step), dtype=torch.float, device=self.device)
        # TODO: This is currently set up only for num_obs_steps = 2!!
        self._curr_obs_buf = torch.zeros_like(self._obs_buf[:, 0])
        self._past_obs_buf = torch.zeros_like(self._obs_buf[:, 1])


    def post_step_procedures(self, newobs, indices=None):
        """
        Post step procedures to compute additional quantities needed by any algos. Pairs observation seen in rollouts to get s-s' vectors

        Computes the observation buffer needed by amp_continuous and similar algos

        Args:
            newobs (np array): New observations seen after stepping the env
            indices (list): Indices of environments that were stepped (default: None)
        """
        if indices is None:
            self._curr_obs_buf[:] = torch.from_numpy(newobs).to(self.device)
            self._obs_buf[:, 1] = copy.deepcopy(self._curr_obs_buf)
            self._obs_buf[:, 0] = copy.deepcopy(self._past_obs_buf)
            # self._obs_buf[:, 0] = torch.from_numpy(np.concatenate((self._past_obs_buf, newobs), axis=1))
            self._past_obs_buf[:] = torch.from_numpy(newobs).to(self.device)
            self._curr_obs_buf[:] = torch.zeros_like(self._obs_buf[:, 0], device=self.device)
        else:
            self._curr_obs_buf[indices] = torch.from_numpy(newobs).to(self.device)
            self._obs_buf[:, 1] = copy.deepcopy(self._curr_obs_buf)
            self._obs_buf[:, 0] = copy.deepcopy(self._past_obs_buf)
            self._past_obs_buf[indices] = torch.from_numpy(newobs).to(self.device)
            self._curr_obs_buf[:] = torch.zeros_like(self._obs_buf[:, 0], device=self.device)


    def augment_infos(self, infos, dones, indices=None):
        """Augment the infos dictionary returned by the step functions

        Args:
            infos (dict): The unmodified infos dict
            dones (list): list of environment indices that were done)
            indices (list): list of environment indices that were stepped (default: None)
        """
        if indices is None:
            paired_obs= self._obs_buf.view(-1, self.num_obs)
            # amp_obs and paired_obs contain the same observations
            # amp_obs is added to maintain compatibility with adversatial motion priors
            infos["amp_obs"] = paired_obs
            infos["paired_obs"] = paired_obs
            infos["terminate"] = torch.from_numpy(self.concat_func(dones)).to(self.device)
            return infos

        else:
            paired_obs= self._obs_buf.view(-1, self.num_obs)
            paired_obs = paired_obs[indices]
            if len(indices) == 1:
                infos = infos[0]
            infos["amp_obs"] = paired_obs
            infos["paired_obs"] = paired_obs
            infos["terminate"] = torch.from_numpy(self.concat_func(dones)).to(self.device)
            infos["stepped_indices"] = indices
            return infos


    def render(self):
        """
        Render one (first) of the parallel environments
        """
        # Rendering
        if not self.headless:
            res = self.workers[0].render.remote()
            _ = self.ray.get(res)

            # time.sleep(0.04) # 50 fps = 0.08s wait