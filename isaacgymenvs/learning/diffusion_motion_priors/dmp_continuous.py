import copy
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
import yaml
import argparse

from rl_games.algos_torch import a2c_continuous
# from rl_games.algos_torch import torch_ext
# from rl_games.algos_torch import central_value
# from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
# from rl_games.common import datasets
# from rl_games.common import schedulers
# from rl_games.common import vecenv

import torch
from torch import optim

# from . import amp_datasets as amp_datasets

from tensorboardX import SummaryWriter
from learning.motion_ncsn.models.motion_scorenet import SimpleNet

def dict2namespace(config):
    """Convert a disctionary (typically containing config params) to a namespace structure (https://tedboy.github.io/python_stdlib/generated/generated/argparse.Namespace.html#argparse.Namespace)

    Args:
        config (dict): dictionary of configs params
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

class DMPAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, params):
        """Initialise DMP algorithm with passed params. Inherit from the rl_games PPO implementation.

        Args:
            base_name (:obj:`str`): Name passed on to the observer and used for checkpoints etc.
            params (:obj `dict`): Algorithm parameters

        """
        super().__init__(base_name, params)
        config = params['config']
        self._load_config_params(config)
        self._init_network(config['dmp_config'])

        # if self.normalize_value:
        #     self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std
        # if self._normalize_energynet_input:
            # self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)

        print("This is DMP")


    def _load_config_params(self, config):
        """Load algorithm parameters passed via the config file

        Args:
            config (dict): Configuration params
        """
        
        self._task_reward_w = config['dmp_config']['inference']['task_reward_w']
        self._energy_reward_w = config['dmp_config']['inference']['energy_reward_w']

        self._paired_observation_space = self.env_info['paired_observation_space']
        self._eb_model_checkpoint = config['dmp_config']['inference']['eb_model_checkpoint']
        self._c = config['dmp_config']['inference']['sigma_level'] # c ranges from [0,L-1]
        self._sigma_begin = config['dmp_config']['model']['sigma_begin']
        self._sigma_end = config['dmp_config']['model']['sigma_end']
        self._L = config['dmp_config']['model']['L']
        self._normalize_energynet_input = config['dmp_config']['training'].get('normalize_energynet_input', True)


    def init_tensors(self):
        """Initialise the default experience buffer (used in PPO in rl_games) and add additional tensors to track
        """

        super().init_tensors()
        self._build_buffers()


    def _init_network(self, energynet_config):
        """Initialise the energy-based model based on the parameters in the config file

        Args:
            energynet_config (dict): Configuration parameters used to define the energy network
        """

        # Convert to Namespace() 
        energynet_config = dict2namespace(energynet_config)

        eb_model_states = torch.load(self._eb_model_checkpoint, map_location=self.ppo_device)
        energynet = SimpleNet(energynet_config).to(self.ppo_device)
        energynet = torch.nn.DataParallel(energynet)
        energynet.load_state_dict(eb_model_states[0])

        self._energynet = energynet
        self._energynet.eval()
        self._sigmas = np.exp(np.linspace(np.log(self._sigma_begin), np.log(self._sigma_end), self._L))


    def _build_buffers(self):
        """Set up the experience buffer to track tensors required by the algorithm.

        Here, paired_obs tracks a set of s-s' pairs used to compute energies. Refer to rl_games.common.a2c_common and rl_games.common.experience for more info
        """

        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['paired_obs'] = torch.zeros(batch_shape + self._paired_observation_space.shape,
                                                                    device=self.ppo_device)
        

    def _env_reset_done(self):
        """Reset any environments that are in the done state. 
        
        Wrapper around the vec_env reset_done() method. Internally, it handles several cases of envs being done (all done, no done, some done etc.)
        """

        obs, done_env_ids = self.vec_env.reset_done()
        return self.obs_to_tensors(obs), done_env_ids


    def _calc_rewards(self, paired_obs):
        """Calculate DMP rewards given a sest of observation pairs

        Args:
            paired_obs (torch.Tensor): A pair of s-s' observations (usually extracted from the replay buffer)
        """

        energy_rew = self._calc_energy(paired_obs)
        output = {
            'energy_reward': energy_rew
        }

        return output

    def _calc_energy(self, paired_obs):
        """Run the pre-trained energy-based model to compute rewards as energies

        Args:
            paired_obs (torch.Tensor): A pair of s-s' observations (usually extracted from the replay buffer)
        """
        # if self._normalize_energynet_input:
        #     pass

        ### TESTING - ONLY FOR PARTICLE ENV 2D ###
        paired_obs = paired_obs[:,:,:2]
        ### TESTING - ONLY FOR PARTICLE ENV 2D ###

        # Reshape from being (horizon_len, num_envs, paired_obs_shape) to (-1, paired_obs_shape)
        original_shape = list(paired_obs.shape)
        paired_obs = paired_obs.reshape(-1, original_shape[-1])

        # Tensor of noise level to condition the energynet
        labels = torch.ones(paired_obs.shape[0], device=paired_obs.device) * self._c # c ranges from [0,L-1]
        
        with torch.no_grad():
            energy_rew = self._energynet(paired_obs, labels)
            original_shape[-1] = energy_rew.shape[-1]
            energy_rew = energy_rew.reshape(original_shape)

        return energy_rew

    def _combine_rewards(self, task_rewards, dmp_rewards):
        """Combine task and style (energy) rewards using the weights assigned in the config file

        Args:
            task_rewards (torch.Tensor): rewards received from the environment
            dmp_rewards (torch.Tensor): rewards obtained as energies computed using an energy-based model
        """

        energy_rew = dmp_rewards['energy_reward']
        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._energy_reward_w * energy_rew
        return combined_rewards


    # def _preproc_obs(self, obs):
    #     """Preprocess observations (normalization)

    #     Args:
    #         obs (torch.Tensor): observations to feed into the energy-based model
    #     """

    #     if self._normalize_amp_input:
    #         obs = self._amp_input_mean_std(obs)
    #     return obs


    def play_steps(self):
        """Rollout the current policy for some horizon length to obtain experience samples (s, s', r, info). 
        
        Also compute augmented rewards and save for later optimisation 
        """

        self.set_eval()

        print("This is the play_steps method modified for DMP")
        update_list = self.update_list
        step_time = 0.0

        for n in range(self.horizon_length):
            ## New Addition ##
            # Reset the environments that are in the done state. Needed to get the initial observation of the paired observation.
            self.obs, done_env_ids = self._env_reset_done()
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            
            # self.experience_buffer.update_data('obses', n, self.obs['obs'])
            # self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('dones', n, self.dones)
            ## New Addition ##
            self.experience_buffer.update_data('paired_obs', n, infos['paired_obs'])

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]
     
            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            ## New Addition ##
            # if (self.vec_env.env.viewer and (n == (self.horizon_length - 1))):
            #     self._print_debug_stats(infos)

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']

        ## New Addition ##
        mb_paired_obs = self.experience_buffer.tensor_dict['paired_obs']

        ## New Addition ##
        dmp_rewards = self._calc_rewards(mb_paired_obs)
        mb_rewards = self._combine_rewards(mb_rewards, dmp_rewards)

        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        ## New Addition ##
        # Adding dmp rewards to batch dict. Not used anywhere. Can be accessed in a network update later if needed
        # for k, v in amp_rewards.items():
        #     batch_dict[k] = a2c_common.swap_and_flatten01(v)
        quit()
        return batch_dict


