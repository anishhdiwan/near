import copy
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
import yaml

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import torch_ext
# from rl_games.algos_torch import central_value
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
# from rl_games.common import datasets
# from rl_games.common import schedulers
# from rl_games.common import vecenv

import torch
from torch import optim

# from . import amp_datasets as amp_datasets

from tensorboardX import SummaryWriter
from learning.motion_ncsn.models.motion_scorenet import SimpleNet
from utils.ncsn_utils import dict2namespace

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

        # Standardization
        if self._normalize_energynet_input:
            ## TESTING ONLY: Swiss-Roll ##
            # self._energynet_input_norm = RunningMeanStd(torch.ones(config['dmp_config']['model']['in_dim']).shape).to(self.ppo_device)
            ## TESTING ONLY ##

            self._energynet_input_norm = RunningMeanStd(self._paired_observation_space.shape).to(self.ppo_device)
            # Since the running mean and std are pre-computed on the demo data, only eval is needed here

            energynet_input_norm_states = torch.load(self._energynet_input_norm_checkpoint, map_location=self.ppo_device)
            self._energynet_input_norm.load_state_dict(energynet_input_norm_states)

            self._energynet_input_norm.eval()

        print("Diffusion Motion Priors Initialised!")


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
        # self._sigmas = np.exp(np.linspace(np.log(self._sigma_begin), np.log(self._sigma_end), self._L))


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


    def _preprocess_observations(self, obs):
        """Preprocess observations (normalization)

        Args:
            obs (torch.Tensor): observations to feed into the energy-based model
        """
        if self._normalize_energynet_input:
            obs = self._energynet_input_norm(obs)
        return obs


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
            if (self.vec_env.env.viewer and (n == (self.horizon_length - 1))):
                self._print_debug_stats(infos)

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
        # Adding dmp rewards to batch dict. Used for early stopping and reward logging
        for k, v in dmp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)
        batch_dict['combined_rewards'] = a2c_common.swap_and_flatten01(mb_rewards)

        return batch_dict


    def _print_debug_stats(self, infos):
        """Print training stats for debugging. Usually called at the end of every training epoch

        Args:
            infos (dict): Dictionary containing infos passed to the algorithms after stepping the environment
        """

        # TODO: Handle the case where paired obs is not provided (isaacgym envs)
        paired_obs = infos['paired_obs']

        shape = list(paired_obs.shape)
        shape.insert(0,1)
        paired_obs = paired_obs.view(shape)
        energy_rew = self._calc_energy(paired_obs)

        print(f"Mean energy reward (across all envs): {energy_rew.mean()}")

    def _log_train_info(self, infos, frame):
        """Log dmp specific training information

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
            energy_rew, combined_rewards, step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            ## New Addition ##
            dmp_infos = {'energy_reward': energy_rew, 'combined_rewards': combined_rewards}
            mean_combined_reward = round(torch.mean(combined_rewards).item(), ndigits=4)

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False

            if self.rank == 0:
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
                self._log_train_info(dmp_infos, frame)
                
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

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_combined_rew_' + str(mean_combined_reward)

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_combined_reward <= self.last_mean_rewards):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

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

        ## New Addition ##
        energy_rew = batch_dict.get('energy_reward', torch.zeros(self.num_agents))
        combined_rewards = batch_dict.get('combined_rewards', torch.zeros(self.num_agents))

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

        return energy_rew, combined_rewards, batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul