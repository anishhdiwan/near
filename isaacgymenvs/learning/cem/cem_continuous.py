import copy
import os
from math import floor

from rl_games.common import vecenv


from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from utils.ncsn_utils import dict2namespace
from learning.motion_ncsn.models.motion_scorenet import SimpleNet

# from rl_games.common.experience import ExperienceBuffer

import numpy as np
import time
import gym

from datetime import datetime
from tensorboardX import SummaryWriter
import torch 
# from torch import nn
# import torch.distributed as dist
 
# from time import sleep

# from rl_games.common import common_losses

from abc import ABC
from abc import abstractmethod, abstractproperty

class BaseAlgorithm(ABC):
    def __init__(self, base_name, config):
        pass

    @abstractproperty
    def device(self):
        pass

    # @abstractmethod
    # def clear_stats(self):
    #     pass

    @abstractmethod
    def train(self):
        pass

    # @abstractmethod
    # def train_epoch(self):
    #     pass

    # @abstractmethod
    # def get_full_state_weights(self):
    #     pass

    # @abstractmethod
    # def set_full_state_weights(self, weights, set_epoch):
    #     pass

    # @abstractmethod
    # def get_weights(self):
    #     pass

    # @abstractmethod
    # def set_weights(self, weights):
    #     pass

    # # Get algo training parameters
    # @abstractmethod
    # def get_param(self, param_name):
    #     pass

    # # Set algo training parameters
    # @abstractmethod
    # def set_param(self, param_name, param_value):
    #     pass


class CEMAgent(BaseAlgorithm):

    def __init__(self, base_name, params):
        print("Init CEM!!")
        self.config = config = params['config']
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + datetime.now().strftime("_%d-%H-%M-%S")

        self.config = config

        self.algo_device = config.get('device', 'cuda:0')
        self.curr_frames = 0
        self.log_path = config.get('log_path', "runs/")

        self.env_config = config.get('env_config', {})
        self.num_actors = config['num_actors']
        self.env_name = config['env_name']

        self.vec_env = None
        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()
        else:
            self.vec_env = config.get('vec_env', None)


        self.observation_space = self.env_info['observation_space']

        # self.is_train = config.get('is_train', True)


        self.save_freq = config.get('save_frequency', 0)
        self.save_best_after = config.get('save_best_after', 100)
        self.print_stats = config.get('print_stats', True)
        self.name = base_name

        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        self.num_sims = self.config['num_sims']
        self.elite_percentage = self.config['elite_percentage']



        self.rewards_shaper = config['reward_shaper']
        self.num_agents = self.env_info.get('agents', 1)
        self.horizon_length = config['horizon_length']
        self.seq_len = self.config.get('seq_length', 4)

        self.normalize_input = self.config['normalize_input']


        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k,v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
 

        self.gamma = self.config['gamma']

        self.games_to_track = self.config.get('games_to_track', 100)

        self.game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self.algo_device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.algo_device)
        self.obs = None

        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors
        assert(('minibatch_size_per_env' in self.config) or ('minibatch_size' in self.config))
        self.minibatch_size_per_env = self.config.get('minibatch_size_per_env', 0)
        self.minibatch_size = self.config.get('minibatch_size', self.num_actors * self.minibatch_size_per_env)
        self.mini_epochs_num = self.config['mini_epochs']
        self.num_minibatches = self.batch_size // self.minibatch_size
        assert(self.batch_size % self.minibatch_size == 0)

        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -100500
        self.play_time = 0
        self.epoch_num = 0
        self.curr_frames = 0
        # allows us to specify a folder where all experiments will reside
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)


        self.writer = SummaryWriter(self.summaries_dir)


        self.is_tensor_obses = False
        self.last_state_indices = None


        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.algo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.algo_device)


        # Initialising the energy network
        self._load_config_params(config)
        self._init_network(config['dmp_config'])


    @property
    def device(self):
        return self.algo_device


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
        self._energynet_input_norm_checkpoint = config['dmp_config']['inference']['running_mean_std_checkpoint']

        self.mean_game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self.algo_device)
        self.mean_shaped_task_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self.algo_device)
        self.mean_energy_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self.algo_device)
        self.mean_combined_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self.algo_device)


    def _init_network(self, energynet_config):
        """Initialise the energy-based model based on the parameters in the config file

        Args:
            energynet_config (dict): Configuration parameters used to define the energy network
        """

        # Standardization
        if self._normalize_energynet_input:
            ## TESTING ONLY: Swiss-Roll ##
            # self._energynet_input_norm = RunningMeanStd(torch.ones(config['dmp_config']['model']['in_dim']).shape).to(self.algo_device)
            ## TESTING ONLY ##

            self._energynet_input_norm = RunningMeanStd(self._paired_observation_space.shape).to(self.algo_device)
            # Since the running mean and std are pre-computed on the demo data, only eval is needed here

            energynet_input_norm_states = torch.load(self._energynet_input_norm_checkpoint, map_location=self.algo_device)
            self._energynet_input_norm.load_state_dict(energynet_input_norm_states)

            self._energynet_input_norm.eval()


        # Convert to Namespace() 
        energynet_config = dict2namespace(energynet_config)

        eb_model_states = torch.load(self._eb_model_checkpoint, map_location=self.algo_device)
        energynet = SimpleNet(energynet_config).to(self.algo_device)
        energynet = torch.nn.DataParallel(energynet)
        energynet.load_state_dict(eb_model_states[0])

        self._energynet = energynet
        self._energynet.eval()
        # self._sigmas = np.exp(np.linspace(np.log(self._sigma_begin), np.log(self._sigma_end), self._L))


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

        # ### TESTING - ONLY FOR PARTICLE ENV 2D ###
        # paired_obs = paired_obs[:,:,:2]
        # ### TESTING - ONLY FOR PARTICLE ENV 2D ###
        paired_obs = self._preprocess_observations(paired_obs)
        # Reshape from being (horizon_len, num_envs, paired_obs_shape) to (-1, paired_obs_shape)
        original_shape = list(paired_obs.shape)
        paired_obs = paired_obs.reshape(-1, original_shape[-1])

        # Tensor of noise level to condition the energynet
        labels = torch.ones(paired_obs.shape[0], device=paired_obs.device) * self._c # c ranges from [0,L-1]
        
        with torch.no_grad():
            energy_rew = -self._energynet(paired_obs, labels)
            original_shape[-1] = energy_rew.shape[-1]
            energy_rew = energy_rew.reshape(original_shape)

        return energy_rew


    def _combine_rewards(self, task_rewards, energy_rewards):
        """Combine task and style (energy) rewards using the weights assigned in the config file

        Args:
            task_rewards (torch.Tensor): rewards received from the environment
            energy_rewards (torch.Tensor): rewards obtained as energies computed using an energy-based model
        """

        energy_rew = energy_rewards['energy_reward']
        combined_rewards = (self._task_reward_w * task_rewards) + (self._energy_reward_w * energy_rew)
        return combined_rewards


    def _preprocess_observations(self, obs):
        """Preprocess observations (normalization)

        Args:
            obs (torch.Tensor): observations to feed into the energy-based model
        """
        if self._normalize_energynet_input:
            obs = self._energynet_input_norm(obs)
        return obs



    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def init_sampling_distributions(self):
        """Initialise the CEM action sampling distributions
        """
        self.action_mu = torch.zeros(self.horizon_length, self.actions_num, device=self.algo_device)
        self.action_std = torch.ones(self.horizon_length, self.actions_num, device=self.algo_device)


    def env_step(self, actions):
        """Step the simulated environments
        """
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step_selected(indices=[0], actions=actions)

        if self.is_tensor_obses:
            rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.algo_device), dones.to(self.algo_device), infos
        else:
            rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.algo_device).float(), torch.from_numpy(dones).to(self.algo_device), infos

    def env_reset(self):
        """
        Reset the environment
        """
        obs = self.vec_env.reset_selected(indices=[0])
        obs = self.obs_to_tensors(obs)
        return obs

    def sim_reset(self, state):
        """Reset the simulation environments to a given state

        Args:
            state (gym.Box): The state to reset to
        """
        obs = self.vec_env.reset_selected(indices=list(range(1, self.num_actors)), state=state)
        obs = self.obs_to_tensors(obs)
        return obs


    def sim_step(self, actions):
        """Step the simulated environments

        Args:
            actions (tensor): actions to be taken in the envs
        """
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step_selected(indices=list(range(1, self.num_actors)), actions=actions)

        if self.is_tensor_obses:
            rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.algo_device), dones.to(self.algo_device), infos
        else:
            rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.algo_device).float(), torch.from_numpy(dones).to(self.algo_device), infos


    def plan(self, curr_state):
        """Plan from the agent's current state to update the action sampling distributions over the horizon. Sample the first action from the updated distribution

        Args:
            curr_state (gym.Box): The current state of the agent from where to plan
        """
        self.init_sampling_distributions()
        self.play_steps(curr_state)

        # Apply the first action using the computed distributions. action = N(a_mu_0, a_std_0)
        action = self.action_mu[0] + self.action_std[0] * torch.randn(self.actions_num, device=self.algo_device).view(1,-1)

        return action

    def play_steps(self, curr_state):
        """Rollout actions to obtain experience to then update the action sampling distribution 

        Args:
            curr_state (np array): The real agent's current state
        """
        num_iters = self.num_sims // self.num_actors

        for _ in range(num_iters):
            sim_obs = self.sim_reset(self.obs_to_np(curr_state))

            # Actions shape = [horizon length, num actors - 1, action dim]
            actions = (self.action_mu.repeat(self.num_actors-1,1,1) + (self.action_std * torch.randn(self.num_actors-1, self.horizon_length, self.actions_num, device=self.algo_device))).view(self.horizon_length, self.num_actors-1, -1)

            for n in range(self.horizon_length):
                # Get action at step n of horizon
                action = copy.deepcopy(actions[n])

                # Step simulated envs
                sim_obs, rewards, self.dones, infos = self.sim_step(action)

                shaped_rewards = self.rewards_shaper(rewards)

                # Compute energy and combined rewards using stepped obs
                energy_rewards = self._calc_rewards(infos['paired_obs'])
                sim_rewards = self._combine_rewards(shaped_rewards, energy_rewards)

                # Compute the elite reward cutoff and mask
                num_elite = floor(self.elite_percentage * len(infos["stepped_indices"]))
                cutoff = torch.sort(sim_rewards, dim=0, descending=True).values[num_elite]
                mask = sim_rewards.squeeze() >= cutoff

                # Update the action distribution of step n 
                self.action_mu[n] = torch.mean(action[mask], dim=0)
                self.action_std[n] = torch.std(action[mask], dim=0)



    def train(self):
        """ "Train" with the cem algorithm to solve the given environment. 
        
        The first environment in the set of ray environments is the "real" agent's environment while the others are simulation copies
        """
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs


        while True:
            action = self.plan(self.obs["obs"][0])
            self.obs, rewards, self.dones, infos = self.env_step(action)

            shaped_rewards = self.rewards_shaper(rewards)
            # Compute energy and combined rewards using stepped obs
            energy_rewards = self._calc_rewards(infos['paired_obs'])
            combined_rewards = self._combine_rewards(shaped_rewards, energy_rewards)

            self.mean_game_rewards.update(rewards.sum(dim=0))
            self.mean_shaped_task_rewards.update(shaped_rewards.sum(dim=0))
            self.mean_energy_rewards.update(energy_rewards["energy_reward"].sum(dim=0))
            self.mean_combined_rewards.update(combined_rewards.sum(dim=0))
            
            frame = self.frame // self.num_agents

            curr_frames = self.curr_frames
            self.frame += curr_frames

            print(f"Frames (of simulations): {frame}")


            if self.mean_game_rewards.current_size > 0:
                mean_rewards = self.mean_game_rewards.get_mean()
                mean_shaped_rewards = self.mean_shaped_task_rewards.get_mean()
                mean_energy_rewards = self.mean_energy_rewards.get_mean()
                mean_combined_rewards = self.mean_combined_rewards.get_mean()


                self.writer.add_scalar('game_reward/step', mean_rewards, frame)
                self.writer.add_scalar('combined_reward/step', mean_combined_rewards, frame)
                self.writer.add_scalar('shaped_task_reward/step', mean_shaped_rewards, frame)
                self.writer.add_scalar('energy_reward/step', mean_energy_rewards, frame)


    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert(obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.algo_device)
            else:
                obs = torch.FloatTensor(obs).to(self.algo_device)
        return obs

    def obs_to_tensors(self, obs):
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or 'obs' not in obs:    
            upd_obs = {'obs' : upd_obs}
        return upd_obs

    def obs_to_np(self, obs):
        if isinstance(obs, torch.Tensor):
            return obs.cpu().detach().numpy()
        elif isinstance(obs, np.ndarray):
            return obs


    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

