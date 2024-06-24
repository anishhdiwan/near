import copy
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
import yaml
import random
from math import floor

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import torch_ext
# from rl_games.algos_torch import central_value
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
# from rl_games.common import datasets
# from rl_games.common import schedulers
from rl_games.common import vecenv

import torch
from torch import optim
from tslearn.metrics import dtw as ts_dtw
# from . import amp_datasets as amp_datasets

from tensorboardX import SummaryWriter
from learning.motion_ncsn.models.motion_scorenet import SimpleNet, SinusoidalPosEmb
from utils.ncsn_utils import dict2namespace, LastKMovingAvg, get_series_derivative, to_relative_pose

# tslearn throws numpy deprecation warnings because of version mismatch. Silencing for now
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class NEARAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, params):
        """Initialise NEAR algorithm with passed params. Inherit from the rl_games PPO implementation.

        Args:
            base_name (:obj:`str`): Name passed on to the observer and used for checkpoints etc.
            params (:obj `dict`): Algorithm parameters

        """
        config = params['config']

        # If using temporal feature in the state vector, create the environment first and then augment the env_info to account for extra dims
        if config['near_config']['model']['encode_temporal_feature']:
            print("Using Temporal Features")
            env_config = config.get('env_config', {})
            num_actors = config['num_actors']
            env_name = config['env_name']
            temporal_emb_dim = config['near_config']['model'].get('temporal_emb_dim', None)
            assert temporal_emb_dim != None, "A temporal embedding dim must be provided if encoding temporal features"

            vec_env = vecenv.create_vec_env(env_name, num_actors, **env_config)
            self.env_info = vec_env.get_env_info(temporal_feature=True, temporal_emb_dim=temporal_emb_dim)
            params['config']['env_info'] = self.env_info

        super().__init__(base_name, params)

        # Set the self.vec_env attribute
        if config['near_config']['model']['encode_temporal_feature']:
            self.vec_env = vec_env


        self._load_config_params(config)
        self._init_network(config['near_config'])

        # Standardization
        if self._normalize_energynet_input:
            ## TESTING ONLY: Swiss-Roll ##
            # self._energynet_input_norm = RunningMeanStd(torch.ones(config['near_config']['model']['in_dim']).shape).to(self.ppo_device)
            ## TESTING ONLY ##

            self._energynet_input_norm = RunningMeanStd(self._paired_observation_space.shape).to(self.ppo_device)

            # Since the running mean and std are pre-computed on the demo data, only eval is needed here

            energynet_input_norm_states = torch.load(self._energynet_input_norm_checkpoint, map_location=self.ppo_device)
            self._energynet_input_norm.load_state_dict(energynet_input_norm_states)

            self._energynet_input_norm.eval()
        
        # Fetch demo trajectories for computing eval metrics
        self._fetch_demo_dataset()
        self.sim_asset_root_body_id = None 
        print("Noise-conditioned Energy-based Annealed Rewards Initialised!")


    def _load_config_params(self, config):
        """Load algorithm parameters passed via the config file

        Args:
            config (dict): Configuration params
        """
        
        self._task_reward_w = config['near_config']['inference']['task_reward_w']
        self._energy_reward_w = config['near_config']['inference']['energy_reward_w']

        self.perf_metrics_freq = config['near_config']['inference'].get('perf_metrics_freq', 0)
        self._paired_observation_space = self.env_info['paired_observation_space']
        self._eb_model_checkpoint = config['near_config']['inference']['eb_model_checkpoint']
        self._c = config['near_config']['inference']['sigma_level'] # c ranges from [0,L-1] or is equal to -1
        if self._c == -1:
            # When c=-1, noise level annealing is used.
            print("Using Annealed Rewards!")
            self.ncsn_annealing = True
            
            # Index of the current noise level used to compute energies
            self._c_idx = 0
            # Value to add to reward to ensure that the annealed rewards are non-decreasing
            self._curr_reward_offset = 0.0
            # Sigma levels to use to compute energies
            self._anneal_levels = [3,4,5,6,7,8,9]
            # Noise level being used right now
            self._c = self._anneal_levels[self._c_idx]
            # Minimum energy value after which noise level is changed
            self._anneal_threshold = 100.0 - self._c * 10
            # Initialise a replay memory style class to return the average energy of the last k policies (for both noise levels)
            self._nextlv_energy_buffer = LastKMovingAvg()
            self._thislv_energy_buffer = LastKMovingAvg()
            # Initialise a replay memory style class to return the average reward encounter by the last k policies
            self._transformed_rewards_buffer = LastKMovingAvg()

        else:
            self.ncsn_annealing = False
            self._curr_reward_offset = 0.0
            # Initialise a replay memory style class to return the average reward encounter by the last k policies
            self._transformed_rewards_buffer = LastKMovingAvg()

        self._sigma_begin = config['near_config']['model']['sigma_begin']
        self._sigma_end = config['near_config']['model']['sigma_end']
        self._L = config['near_config']['model']['L']
        self._normalize_energynet_input = config['near_config']['training'].get('normalize_energynet_input', True)
        self._energynet_input_norm_checkpoint = config['near_config']['inference']['running_mean_std_checkpoint']
        
        # If temporal features are encoded in the paired observations then a new space for the temporal states is made. The energy net and normalization use this space
        self._encode_temporal_feature = config['near_config']['model']['encode_temporal_feature']

        try:
            self._max_episode_length = self.vec_env.env.max_episode_length
        except AttributeError as e:
            self._max_episode_length = None
        
        if self._encode_temporal_feature:
            assert self._max_episode_length != None, "A max episode length must be known when using temporal state features"

            # Positional embedding for temporal information
            self.emb_dim = config['near_config']['model']['temporal_emb_dim']
            self.embed = SinusoidalPosEmb(dim=self.emb_dim, steps=512)
            self.embed.eval()

        


    def init_tensors(self):
        """Initialise the default experience buffer (used in PPO in rl_games) and add additional tensors to track
        """

        super().init_tensors()
        self._build_buffers()
        self.mean_shaped_task_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.mean_energy_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.mean_combined_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.mean_energy = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)


    def _init_network(self, energynet_config):
        """Initialise the energy-based model based on the parameters in the config file

        Args:
            energynet_config (dict): Configuration parameters used to define the energy network
        """

        # Convert to Namespace() 
        energynet_config = dict2namespace(energynet_config)

        eb_model_states = torch.load(self._eb_model_checkpoint, map_location=self.ppo_device)
        energynet = SimpleNet(energynet_config, in_dim=self._paired_observation_space.shape[0]).to(self.ppo_device)
        energynet = torch.nn.DataParallel(energynet)
        energynet.load_state_dict(eb_model_states[0])

        self._energynet = energynet
        self._energynet.eval()


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

    
    def _env_reset_all(self):
        """Reset all environments regardless of the done state
        
        Wrapper around the vec_env reset_all() method
        """

        obs = self.vec_env.env.reset_all()
        return self.obs_to_tensors(obs)


    def _calc_rewards(self, paired_obs):
        """Calculate NEAR rewards given a sest of observation pairs

        Args:
            paired_obs (torch.Tensor): A pair of s-s' observations (usually extracted from the replay buffer)
        """

        # energy_rew = - self._calc_energy(paired_obs)
        energy_rew = self._transform_rewards(self._calc_energy(paired_obs))
        output = {
            'energy_reward': energy_rew
        }

        return output

    def _transform_rewards(self, rewards):
        """ Apply a transformation function to the rewards (scale, clip, and offset rewards in desirable range)

        Args:
            rewards (torch.Tensor): Rewards to transform
        """

        # rewards = (0.0001/(1 + rewards**2)) + (torch.exp(-torch.abs(rewards)))
        # rewards = -torch.log(1/(1 + torch.exp(rewards)))

        rewards = rewards + self._curr_reward_offset

        # Keep track of the energy as learning progresses
        self.mean_energy.update(rewards.sum(dim=0))

        # Update the reward transformation once every few frames
        if (self.epoch_num % 3==0) or (self.epoch_num==1):
            self.mean_offset_rew = self._transformed_rewards_buffer.append(rewards)
        else:
            self._transformed_rewards_buffer.append(rewards, return_avg=False)

        # Reward = tanh((reward - avg. reward of last k policies)/10)
        rewards = torch.tanh((rewards - self.mean_offset_rew)/10) # 10 tanh((x - mean)/10) Rewards range between +/-1 . Division by 10 expands the range of non-asympotic inputs

        return rewards


    def _calc_energy(self, paired_obs, c=None):
        """Run the pre-trained energy-based model to compute rewards as energies. 
        
        If a noise level is passed in then the energy for that level is returned. Else the current noise_level is used

        Args:
            paired_obs (torch.Tensor): A pair of s-s' observations (usually extracted from the replay buffer)
            c (int): Noise level to use to condition the energy based model. If not given then the class variable (self._c) is used
        """

        if c is None:
            c = self._c

        # ### TESTING - ONLY FOR PARTICLE ENV 2D ###
        # paired_obs = paired_obs[:,:,:2]
        # ### TESTING - ONLY FOR PARTICLE ENV 2D ###
        paired_obs = self._preprocess_observations(paired_obs)
        # Reshape from being (horizon_len, num_envs, paired_obs_shape) to (-1, paired_obs_shape)
        original_shape = list(paired_obs.shape)
        paired_obs = paired_obs.reshape(-1, original_shape[-1])

        # Tensor of noise level to condition the energynet
        labels = torch.ones(paired_obs.shape[0], device=paired_obs.device) * c # c ranges from [0,L-1]
        
        with torch.no_grad():
            energy_rew = self._energynet(paired_obs, labels)
            original_shape[-1] = energy_rew.shape[-1]
            energy_rew = energy_rew.reshape(original_shape)

        return energy_rew

    def _combine_rewards(self, task_rewards, near_rewards):
        """Combine task and style (energy) rewards using the weights assigned in the config file

        Args:
            task_rewards (torch.Tensor): rewards received from the environment
            near_rewards (torch.Tensor): rewards obtained as energies computed using an energy-based model
        """

        energy_rew = near_rewards['energy_reward']
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


    def _anneal_noise_level(self, **kwargs):
        """If NCSN annealing is used, change the currently used noise level self._c
        """
        ANNEAL_STRATEGY = "non-decreasing-linear" # options are "linear" or "non-decreasing-linear"
        
        if self.ncsn_annealing == True:
            if ANNEAL_STRATEGY == "linear":
                    max_level_iters = 1e6
                    num_levels = self._L
                    self._c = floor((self.frame * num_levels)/max_level_iters)

            elif ANNEAL_STRATEGY == "non-decreasing-linear":
                # Make sure that an observation pair is passed in 
                assert "paired_obs" in list(kwargs.keys())
                paired_obs = kwargs["paired_obs"]

                # If already at the max noise level, do nothing
                if not self._c_idx == len(self._anneal_levels) - 1:

                    # If the next noise level's average energy is lower than some threshold then keep using the current noise level
                    if self._nextlv_energy_buffer.append(self._calc_energy(paired_obs, c=self._anneal_levels[self._c_idx+1])) < self._anneal_threshold:
                        self._c = self._anneal_levels[self._c_idx]

                        # Computing energies for current level twice (once again during play loop). A bit redundant but done for readability and reusability
                        self._thislv_energy_buffer.append(self._calc_energy(paired_obs, c=self._anneal_levels[self._c_idx]), return_avg=False)
                    # If the next noise level's average energy is higher than some threshold then change the noise level and 
                    # add the average energy of the current noise level to the reward offset. This ensures that rewards are non-decreasing
                    else:
                        self._curr_reward_offset += self._thislv_energy_buffer.append(self._calc_energy(paired_obs, c=self._anneal_levels[self._c_idx]))
                        self._c_idx += 1
                        self._c = self._anneal_levels[self._c_idx]
                        self._anneal_threshold = 100.0 - self._c * 10

                        self._thislv_energy_buffer.reset()
                        self._nextlv_energy_buffer.reset()


    def play_steps(self):
        """Rollout the current policy for some horizon length to obtain experience samples (s, s', r, info). 
        
        Also compute augmented rewards and save for later optimisation 
        """

        self.set_eval()

        update_list = self.update_list
        step_time = 0.0

        for n in range(self.horizon_length):
            ## New Addition ##
            # Reset the environments that are in the done state. Needed to get the initial observation of the paired observation.
            self.obs, done_env_ids = self._env_reset_done()

            # Append temporal feature to observations if needed
            if self._encode_temporal_feature:
                # progress0 = torch.unsqueeze(self.vec_env.env.progress_buf/self._max_episode_length, -1)
                progress0 = self.embed(self.vec_env.env.progress_buf/self._max_episode_length)
                self.obs['obs'] = torch.cat((progress0, self.obs['obs']), -1)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)


            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            # Take an action in each environment
            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()
            step_time += (step_time_end - step_time_start)

            # Append temporal feature to observations if needed
            if self._encode_temporal_feature:
                # progress1 = torch.unsqueeze(self.vec_env.env.progress_buf/self._max_episode_length, -1)
                progress1 = self.embed(self.vec_env.env.progress_buf/self._max_episode_length)
                self.obs['obs'] = torch.cat((progress1, self.obs['obs']), -1)
                
                try:
                    obs1, obs0 = torch.chunk(infos['paired_obs'], chunks=2, dim=-1)
                    obs1 = torch.cat((progress1, obs1), -1)
                    obs0 = torch.cat((progress0, obs0), -1)
                    infos['paired_obs'] = torch.cat((obs1, obs0), -1)
                except KeyError:
                    obs1, obs0 = torch.chunk(infos['amp_obs'], chunks=2, dim=-1)
                    obs1 = torch.cat((progress1,obs1), -1)
                    obs0 = torch.cat((progress0,obs0), -1)
                    infos['amp_obs'] = torch.cat((obs1, obs0), -1)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('dones', n, self.dones)
            ## New Addition ##
            try:
                self.experience_buffer.update_data('paired_obs', n, infos['paired_obs'])
            except KeyError:
                self.experience_buffer.update_data('paired_obs', n, infos['amp_obs'])

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
            if (self.vec_env.env.viewer and (n == (self.horizon_length - 1))):
                self._print_debug_stats(infos)


        last_values = self.get_values(self.obs)
        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']

        shaped_env_rewards = copy.deepcopy(mb_rewards).squeeze()

        ## New Addition ##
        mb_paired_obs = self.experience_buffer.tensor_dict['paired_obs']

        ## New Addition ##
        self._anneal_noise_level(paired_obs=mb_paired_obs)
        near_rewards = self._calc_rewards(mb_paired_obs)
        mb_rewards = self._combine_rewards(mb_rewards, near_rewards)

        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        temp_combined_rewards = copy.deepcopy(mb_rewards).squeeze()
        temp_energy_rewards = copy.deepcopy(near_rewards['energy_reward']).squeeze()
        self.mean_combined_rewards.update(temp_combined_rewards.sum(dim=0))
        self.mean_energy_rewards.update(temp_energy_rewards.sum(dim=0))
        self.mean_shaped_task_rewards.update(shaped_env_rewards.sum(dim=0))


        return batch_dict


    def run_policy(self):
        """With network updates paused, rollout the current policy until the end of the episode to obtain a trajectory of body poses. 
        
        Used to compute performance metrics.
        """
        is_deterministic = True
        max_steps = self._max_episode_length
        pose_trajectory = []
        self.run_pi_dones = None

        self.run_obses = self._env_reset_all()
        pose_trajectory.append(self._fetch_sim_asset_poses())

        for n in range(max_steps):

            # Append temporal feature to observations if needed
            if self._encode_temporal_feature:
                progress0 = self.embed(self.vec_env.env.progress_buf/self._max_episode_length)
                self.run_obses['obs'] = torch.cat((progress0, self.run_obses['obs']), -1)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.run_obses, masks)
            else:
                res_dict = self.get_action_values(self.run_obses)

            if is_deterministic:
                self.run_obses, rewards, dones, infos = self.env_step(res_dict['mus'])
            else:
                self.run_obses, rewards, dones, infos = self.env_step(res_dict['actions'])
                
            pose_trajectory.append(self._fetch_sim_asset_poses())

            all_done_indices = dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]
            done_count = len(env_done_indices)
            
            # Find the envs that were done the last
            if self.run_pi_dones != None:
                new_dones = (dones - self.run_pi_dones).nonzero(as_tuple=False)
            self.run_pi_dones = dones.clone()

            if done_count == self.num_actors:
                # Reset the env to start training again
                self.obs = self._env_reset_all()
                break

        # Select a random env out of those envs that were done last
        env_idx = random.choice(new_dones.squeeze(-1).tolist())
        pose_trajectory = torch.stack(pose_trajectory)
        pose_trajectory = pose_trajectory[:, env_idx, :, : ]
        # Transform to be relative to root body
        root_trajectory = pose_trajectory[:, self.sim_asset_root_body_id, :]
        pose_trajectory = to_relative_pose(pose_trajectory, self.sim_asset_root_body_id)

        return pose_trajectory, root_trajectory



    def _print_debug_stats(self, infos):
        """Print training stats for debugging. Usually called at the end of every training epoch

        Args:
            infos (dict): Dictionary containing infos passed to the algorithms after stepping the environment
        """

        try:
            paired_obs = infos['paired_obs']
        except KeyError:
            paired_obs = infos['amp_obs']

        shape = list(paired_obs.shape)
        shape.insert(0,1)
        paired_obs = paired_obs.view(shape)
        energy_rew = self._calc_energy(paired_obs)

        print(f"Minibatch Stats - E(s_penultimate, s_last).mean(all envs): {energy_rew.mean()}")

    def _log_train_info(self, infos, frame):
        """Log near specific training information

        Args:
            infos (dict): dictionary of training logs
            frame (int): current training step
        """
        for k, v in infos.items():
            self.writer.add_scalar(f'{k}/step', torch.mean(v).item(), frame)

    def train(self):
        """Train the algorithm and log training metrics
        """
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False

            if self.global_rank == 0:
                self.diagnostics.epoch(self, current_epoch = epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames * self.rank_size if self.multi_gpu else self.curr_frames
                self.frame += curr_frames

                a2c_common.print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time, 
                                epoch_num, self.max_epochs, frame, self.max_frames)

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                                a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame,
                                scaled_time, scaled_play_time, curr_frames)

                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

                if self.has_soft_aug:
                    self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

                ## New Addition ##
                if self.mean_combined_rewards.current_size > 0:
                    mean_combined_reward = self.mean_combined_rewards.get_mean()
                    mean_shaped_task_reward = self.mean_shaped_task_rewards.get_mean()
                    mean_energy_reward = self.mean_energy_rewards.get_mean()
                    mean_energy = self.mean_energy.get_mean()

                    self.writer.add_scalar('minibatch_combined_reward/step', mean_combined_reward, frame)
                    self.writer.add_scalar('minibatch_shaped_task_reward/step', mean_shaped_task_reward, frame)
                    self.writer.add_scalar('minibatch_energy_reward/step', mean_energy_reward, frame)
                    self.writer.add_scalar('minibatch_energy/step', mean_energy, frame)
                    self.writer.add_scalar('ncsn_perturbation_level/step', self._c, frame)
                
                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    # checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_combined_rew_' + str(mean_combined_reward)
                    checkpoint_name = f"{self.config['name']}_combined_rew_{str(mean_combined_reward)}_{'{date:%d-%H-%M-%S}'.format(date=datetime.now())}_{str(frame)}"

                    # Compute performance metrics
                    if self.perf_metrics_freq > 0:
                        if (self.epoch_num > 0) and (frame % (self.perf_metrics_freq * self.curr_frames) == 0):
                            self.compute_performance_metrics(frame)

                    if self.save_freq > 0:
                        #  and (mean_combined_reward <= self.last_mean_rewards)
                        if (frame % (self.save_freq * self.curr_frames) == 0):
                            self.save(os.path.join(self.nn_dir, checkpoint_name))

                    if mean_combined_reward > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_combined_reward)
                        self.last_mean_rewards = mean_combined_reward
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                        + '_combined_rew_' + str(mean_combined_reward).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_combined_rew_' + str(mean_combined_reward).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

            if should_exit:
                return self.last_mean_rewards, epoch_num


    def train_epoch(self):
        """Train one epoch of the algorithm
        """
        # super().train_epoch()

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()
        if self.has_central_value:
            self.train_central_value()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)
                if self.schedule_type == 'legacy':
                    av_kls = kl
                    if self.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= self.rank_size
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.rank_size
            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul


    def _fetch_demo_dataset(self):
        """Fetch trajectories of demonstration data where each frame in a trajectory is a vector of cartesian poses of the expert's joints

        trajectories = [traj] where traj = [x0, x1, ..] where xi = [root_pos, joint_pos] 
        """
        demo_trajectories, [self.demo_data_root_body_id, self.demo_data_root_body_name] = self.vec_env.env.fetch_demo_dataset()
        
        # Transform demo_traj to have body poses relative to the individual root pose of the body
        root_trajectories = []
        root_relative_demo_trajectories = []
        for demo_traj in demo_trajectories:
            root_trajectories.append(demo_traj[:,self.demo_data_root_body_id, :])
            root_relative_demo_trajectories.append(to_relative_pose(demo_traj, self.demo_data_root_body_id))
        
        self.demo_trajectories = root_relative_demo_trajectories
        self.demo_root_trajectories = root_trajectories


    def _fetch_sim_asset_poses(self):
        """Fetch the cartesian pose of all joints of the simulation asset in every environment at the current timestep
        """

        # The root body id of the simulation asset
        if self.sim_asset_root_body_id is None: 
            self.sim_asset_root_body_id = self.vec_env.env.body_ids_dict[self.demo_data_root_body_name]
        
        # Shape [num_envs, num_bodies, 3]
        return self.vec_env.env._rigid_body_pos.clone()


    def compute_performance_metrics(self, frame):
        """Compute performance metrics
        """
        self.set_eval()
        agent_pose_trajectory, agent_root_trajectory = self.run_policy()
        num_joints = agent_pose_trajectory.shape[-1]/3
        dt = self.vec_env.env.dt

        # Compute pose error
        dtw_pose_error = 0
        for demo_traj in self.demo_trajectories:
            dtw_pose_error += ts_dtw(demo_traj.clone().cpu(), agent_pose_trajectory.clone().cpu()) / num_joints
        dtw_pose_error = dtw_pose_error/len(self.demo_trajectories)
        self.writer.add_scalar('mean_dtw_pose_error/step', dtw_pose_error, frame)
        print(f"Evaluating current policy's performance. Mean dynamic time warped pose error {dtw_pose_error}")
                
        # Compute root body statistics
        agent_root_velocity = get_series_derivative(agent_root_trajectory, dt)
        agent_root_acceleration = get_series_derivative(agent_root_velocity, dt)
        agent_root_jerk = get_series_derivative(agent_root_acceleration, dt)

        mean_vel_norm = torch.mean(torch.linalg.norm(agent_root_velocity, dim=1))
        mean_acc_norm = torch.mean(torch.linalg.norm(agent_root_acceleration, dim=1))
        mean_jerk_norm = torch.mean(torch.linalg.norm(agent_root_jerk, dim=1))
        self.writer.add_scalar('root_body_velocity/step', mean_vel_norm, frame)
        self.writer.add_scalar('root_body_acceleration/step', mean_acc_norm, frame)
        self.writer.add_scalar('root_body_jerk/step', mean_jerk_norm, frame)

        self.set_train()




        
        