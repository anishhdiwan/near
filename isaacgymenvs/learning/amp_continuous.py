# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv

from isaacgymenvs.utils.torch_jit_utils import to_torch

import time
import copy
from datetime import datetime
import numpy as np
from torch import optim
import torch 
from torch import nn
import random

import isaacgymenvs.learning.replay_buffer as replay_buffer
import isaacgymenvs.learning.common_agent as common_agent 
from utils.ncsn_utils import get_series_derivative, to_relative_pose, sparc
from tslearn.metrics import dtw as ts_dtw

from tensorboardX import SummaryWriter


class AMPAgent(common_agent.CommonAgent):

    def __init__(self, base_name, params):
        super().__init__(base_name, params)

        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std
        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)

        # Fetch demo trajectories for computing eval metrics
        self._fetch_demo_dataset()
        self.sim_asset_root_body_id = None 

        return

    def init_tensors(self):
        super().init_tensors()
        self._build_amp_buffers()

        ## Newly Added ##
        self.mean_shaped_task_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.mean_disc_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.mean_combined_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)

        return
    
    def set_eval(self):
        super().set_eval()
        if self._normalize_amp_input:
            self._amp_input_mean_std.eval()
        return

    def set_train(self):
        super().set_train()
        if self._normalize_amp_input:
            self._amp_input_mean_std.train()
        return

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_amp_input:
            state['amp_input_mean_std'] = self._amp_input_mean_std.state_dict()
        return state

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self._normalize_amp_input:
            self._amp_input_mean_std.load_state_dict(weights['amp_input_mean_std'])
        return

    def play_steps(self):
        self.set_eval()

        epinfos = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs, done_env_ids = self._env_reset_done()
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

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])

            ## TESTING ###
            # print("TESTING")
            # print(f"play_steps amp_obs (same as paired_obs) {infos['amp_obs']}")
            # quit()
            ## TESTING ###

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
        
            if (self.vec_env.env.viewer and (n == (self.horizon_length - 1))):
                self._amp_debug(infos)

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']

        ## New Addition ##
        shaped_env_rewards = copy.deepcopy(mb_rewards).squeeze()

        ## TESTING ###
        # print("TESTING")
        # print(f"play_steps mb_amp_obs {mb_amp_obs.shape}")
        # quit()
        ## TESTING ###

        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        ## New Addition ##
        temp_combined_rewards = copy.deepcopy(mb_rewards).squeeze()
        temp_disc_rewards = copy.deepcopy(amp_rewards['disc_rewards']).squeeze()
        self.mean_combined_rewards.update(temp_combined_rewards.sum(dim=0))
        self.mean_disc_rewards.update(temp_disc_rewards.sum(dim=0))
        self.mean_shaped_task_rewards.update(shaped_env_rewards.sum(dim=0))

        return batch_dict


    def run_policy(self):
        """With network updates paused, rollout the current policy until the end of the episode to obtain a trajectory of body poses. 
        
        Used to compute performance metrics.
        """
        is_deterministic = True
        return_traj_type = "most_rewarding" # "longest" or "most_rewarding"
        most_rewarding_k = 20
        done_envs = None
        max_steps = self._max_episode_length
        pose_trajectory = []
        self.run_pi_dones = None
        self.total_env_learnt_rewards = None
        # cfg_initialization = self.vec_env.env._state_init
        # self.vec_env.env._state_init = self.vec_env.env.StateInit.Uniform
        self.run_obses = self._env_reset_all()
        pose_trajectory.append(self._fetch_sim_asset_poses())

        for n in range(max_steps):

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
            if done_envs is not None:
                done_envs = np.union1d(done_envs, env_done_indices.clone().squeeze().cpu().numpy())
            else:
                done_envs = env_done_indices.clone().squeeze().cpu().numpy()
            
            if return_traj_type == "longest":
                # Find the envs that were done the last
                if self.run_pi_dones is not None:
                    new_dones = (dones - self.run_pi_dones).nonzero(as_tuple=False)
                self.run_pi_dones = dones.clone()

                if done_count == self.num_actors:
                    # Select a random env out of those envs that were done last
                    env_idx = random.choice(new_dones.squeeze(-1).tolist())
                    # Reset the env to start training again
                    # self.vec_env.env._state_init = cfg_initialization
                    self.obs = self._env_reset_all()
                    break

            elif return_traj_type == "most_rewarding":            

                # Compute rews and set done env rews to 0.0
                env_learnt_rewards = self._calc_disc_rewards(infos['amp_obs'])
                env_learnt_rewards[done_envs] = 0.0

                if self.total_env_learnt_rewards is not None:
                    self.total_env_learnt_rewards += env_learnt_rewards.clone()
                else:
                    self.total_env_learnt_rewards = env_learnt_rewards.clone() 

                if done_count == self.num_actors:
                    # Select a random env out of those envs that were done last
                    # env_idx = torch.argmax(self.total_env_learnt_rewards, dim=0).item()
                    _, env_idx = torch.topk(self.total_env_learnt_rewards, k=most_rewarding_k, dim=0)
                    env_idx = env_idx.squeeze().tolist()
                    # Reset the env to start training again
                    # self.vec_env.env._state_init = cfg_initialization
                    self.obs = self._env_reset_all()
                    break


        if isinstance(env_idx, list):
            pose_trajectory = torch.stack(pose_trajectory)
            idx_trajectories = []
            idx_root_trajectories = []
            for i in env_idx:
                idx_pose_trajectory = pose_trajectory[:, i, :, : ]
                # Transform to be relative to root body
                idx_root_trajectory = idx_pose_trajectory[:, self.sim_asset_root_body_id, :]
                idx_pose_trajectory = to_relative_pose(idx_pose_trajectory, self.sim_asset_root_body_id)
                idx_trajectories.append(idx_pose_trajectory)
                idx_root_trajectories.append(idx_root_trajectory)

            return idx_trajectories, idx_root_trajectories

        else:
            pose_trajectory = torch.stack(pose_trajectory)
            pose_trajectory = pose_trajectory[:, env_idx, :, : ]
            # Transform to be relative to root body
            root_trajectory = pose_trajectory[:, self.sim_asset_root_body_id, :]
            pose_trajectory = to_relative_pose(pose_trajectory, self.sim_asset_root_body_id)

            return pose_trajectory, root_trajectory


    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict['amp_obs'] = batch_dict['amp_obs']
        self.dataset.values_dict['amp_obs_demo'] = batch_dict['amp_obs_demo']
        self.dataset.values_dict['amp_obs_replay'] = batch_dict['amp_obs_replay']

        ### TESTING ###
        # print("TESTING")
        # print(f"prepare_dataset amp_obs shape {batch_dict['amp_obs'].shape}")
        # print(f"prepare_dataset amp_obs_demo shape {batch_dict['amp_obs_demo'].shape}")
        # print(f"prepare_dataset amp_obs_replay shape {batch_dict['amp_obs_replay'].shape}")
        # quit()
        ### TESTING ###
        return

    def train_epoch(self):
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps() 

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)
        
        self._update_amp_demos()
        num_obs_samples = batch_dict['amp_obs'].shape[0]        
        amp_obs_demo = self._amp_obs_demo_buffer.sample(num_obs_samples)['amp_obs']
        batch_dict['amp_obs_demo'] = amp_obs_demo

        # TESTING ###
        # print("TESTING")
        # print(f"train_epoch num_obs_samples {num_obs_samples}")
        # print(f"train_epoch amp_obs_demo shape {amp_obs_demo.shape}")
        # quit()
        # TESTING ###

        if (self._amp_replay_buffer.get_total_count() == 0):
            batch_dict['amp_obs_replay'] = batch_dict['amp_obs']
        else:
            batch_dict['amp_obs_replay'] = self._amp_replay_buffer.sample(num_obs_samples)['amp_obs']

        self.set_train()

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])
                
                if self.schedule_type == 'legacy':
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                if (train_info is None):
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)
            
            av_kls = torch_ext.mean_list(train_info['kl'])

            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

        if self.schedule_type == 'standard_epoch':
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        self._store_replay_amp_obs(batch_dict['amp_obs'])

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time
        self._record_train_batch_info(batch_dict, train_info)

        return train_info

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        amp_obs = input_dict['amp_obs'][0:self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)
        amp_obs.requires_grad_(True)
        amp_obs_replay = input_dict['amp_obs_replay'][0:self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        amp_obs_demo = input_dict['amp_obs_demo'][0:self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
            'amp_obs' : amp_obs,
            'amp_obs_replay' : amp_obs_replay,
            'amp_obs_demo' : amp_obs_demo
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            disc_agent_logit = res_dict['disc_agent_logit']
            disc_agent_replay_logit = res_dict['disc_agent_replay_logit']
            disc_demo_logit = res_dict['disc_demo_logit']

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)

            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]
            
            if self.disc_experiment:
                if self.pause_policy_updates and self.disc_expt_reset_disc:
                    # Avoid passing concatenated logits to make sure that discriminator receives the same number of samples from both datasets
                    disc_agent_cat_logit = disc_agent_logit
                else:
                    disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
            else:
                disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
            
            disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo, amp_obs)
            disc_loss = disc_info['disc_loss']

            if self.disc_experiment:
                if self.pause_policy_updates:
                    loss = self._disc_coef * disc_loss
                else:
                    loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                        + self._disc_coef * disc_loss
            else:
                loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                    + self._disc_coef * disc_loss
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()

        # If policy updates are paused, zero the gradients of the policy networks
        if self.disc_experiment:
            if self.pause_policy_updates:
                for child in self.model.named_children():
                    # print(child[0])
                    if child[0] == 'a2c_network':
                        for subchild in child[1].named_children():
                            # print(subchild[0])
                            if subchild[0] in ['_disc_mlp', '_disc_logits']:
                                pass
                            else:
                                for param in subchild[1].parameters():
                                    param.grad = None
                                    param.requires_grad = False

        #TODO: Refactor this ugliest code of the year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
                    
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss,
        }

        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)

        return

    def _load_config_params(self, config):
        super()._load_config_params(config)
        
        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']

        self._amp_observation_space = self.env_info['amp_observation_space']
        self._amp_batch_size = int(config['amp_batch_size'])
        self._amp_minibatch_size = int(config['amp_minibatch_size'])
        assert(self._amp_minibatch_size <= self.minibatch_size)

        self._disc_coef = config['disc_coef']
        self._disc_logit_reg = config['disc_logit_reg']
        self._disc_grad_penalty = config['disc_grad_penalty']
        self._disc_weight_decay = config['disc_weight_decay']
        self._disc_reward_scale = config['disc_reward_scale']
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self.perf_metrics_freq = config.get('perf_metrics_freq', 0)
        self.disc_experiment = config.get('disc_experiment', False)
        if self.disc_experiment:
            self.disc_expt_policy_training = config.get('disc_expt_policy_training', 2e6)
            self.disc_expt_reset_disc = config.get('disc_expt_reset_disc', True)
        self.pause_policy_updates = False

        try:
            self._max_episode_length = self.vec_env.env.max_episode_length
        except AttributeError as e:
            self._max_episode_length = None
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        config['amp_input_shape'] = self._amp_observation_space.shape
        return config

    def _init_train(self):
        super()._init_train()
        self._init_amp_demo_buf()
        return

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo, obs):
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)


        # tracking just the least squares discriminator objective
        disc_loss_least_sq = disc_loss.clone()

        # tracking the gradients of the discriminator w.r.t observations
        grad_disc_obs_agent = torch.autograd.grad(disc_agent_logit, obs, grad_outputs=torch.ones_like(disc_agent_logit), create_graph=False, retain_graph=True)[0].detach().clone()
        grad_disc_obs_demo = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit), create_graph=False, retain_graph=True)[0].detach().clone()
        if self.multi_gpu:
            self.optimizer.zero_grad()
        else:
            for param in self.model.parameters():
                param.grad = None
        grad_disc_obs = torch.cat([grad_disc_obs_agent, grad_disc_obs_demo], dim=0)
        grad_disc_obs = torch.mean(torch.linalg.norm(grad_disc_obs, dim=-1))

        # logit reg
        logit_weights = self.model.a2c_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if (self._disc_weight_decay != 0):
            disc_weights = self.model.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        if self.disc_experiment:
            if self.disc_expt_reset_disc:
                disc_agent_acc, disc_demo_acc, disc_combined_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)
            else:
                disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)
        else:
            disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)


        disc_info = {
            'disc_loss': disc_loss,
            'disc_grad_penalty': disc_grad_penalty,
            'disc_logit_loss': disc_logit_loss,
            'disc_agent_acc': disc_agent_acc,
            'disc_demo_acc': disc_demo_acc,
            'disc_agent_logit': disc_agent_logit,
            'disc_demo_logit': disc_demo_logit,
            'disc_loss_least_sq': disc_loss_least_sq,
            'grad_disc_obs': grad_disc_obs,
        }

        try:
            disc_info['disc_combined_acc'] = disc_combined_acc
        except Exception as e:
            pass

        return disc_info

    def _disc_loss_neg(self, disc_logits):
        if self.disc_experiment:
            if self.disc_expt_reset_disc:
                bce = torch.nn.BCELoss()
            else:
                bce = torch.nn.BCEWithLogitsLoss()
        else:
            bce = torch.nn.BCEWithLogitsLoss()

        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss
    
    def _disc_loss_pos(self, disc_logits):
        if self.disc_experiment:
            if self.disc_expt_reset_disc:
                bce = torch.nn.BCELoss()
            else:
                bce = torch.nn.BCEWithLogitsLoss()
        else:
            bce = torch.nn.BCEWithLogitsLoss()

        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        if self.disc_experiment:
            if self.disc_expt_reset_disc:
                agent_true = disc_agent_logit < 0.5
                agent_acc = torch.mean(agent_true.float())
                demo_true = disc_demo_logit > 0.5
                demo_acc = torch.mean(demo_true.float())
                combined_true = torch.cat([agent_true, demo_true], dim=0)
                combined_acc = torch.mean(combined_true.float())
                return agent_acc, demo_acc, combined_acc
            else:
                agent_acc = disc_agent_logit < 0
                agent_acc = torch.mean(agent_acc.float())
                demo_acc = disc_demo_logit > 0
                demo_acc = torch.mean(demo_acc.float())
                return agent_acc, demo_acc
        else:
            agent_acc = disc_agent_logit < 0
            agent_acc = torch.mean(agent_acc.float())
            demo_acc = disc_demo_logit > 0
            demo_acc = torch.mean(demo_acc.float())
            return agent_acc, demo_acc

    def _fetch_amp_obs_demo(self, num_samples):
        amp_obs_demo = self.vec_env.env.fetch_amp_obs_demo(num_samples)
        return amp_obs_demo

    def _build_amp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['amp_obs'] = torch.zeros(batch_shape + self._amp_observation_space.shape,
                                                                    device=self.ppo_device)
        
        amp_obs_demo_buffer_size = int(self.config['amp_obs_demo_buffer_size'])
        self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)

        self._amp_replay_keep_prob = self.config['amp_replay_keep_prob']
        replay_buffer_size = int(self.config['amp_replay_buffer_size'])
        self._amp_replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size, self.ppo_device)

        self.tensor_list += ['amp_obs']
        return

    def _init_amp_demo_buf(self):
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self._amp_batch_size))

        for i in range(num_batches):
            curr_samples = self._fetch_amp_obs_demo(self._amp_batch_size)
            ### TESTING ###
            # print("TESTING")
            # print(f"init amp demo buffer curr_samples {curr_samples}")
            # print(f"init amp demo buffer curr_samples size {curr_samples.shape}")
            ### TESTING ###
            self._amp_obs_demo_buffer.store({'amp_obs': curr_samples})

        return
    
    def _update_amp_demos(self):
        new_amp_obs_demo = self._fetch_amp_obs_demo(self._amp_batch_size)
        ### TESTING ###
        # print("TESTING")
        # print(f"update_amp_demos new_amp_obs_demo size {new_amp_obs_demo.shape}")
        ### TESTING ###
        self._amp_obs_demo_buffer.store({'amp_obs': new_amp_obs_demo})
        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards['disc_rewards']
        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._disc_reward_w * disc_r
        return combined_rewards

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
            disc_r *= self._disc_reward_scale

        return disc_r

    def _store_replay_amp_obs(self, amp_obs):
        buf_size = self._amp_replay_buffer.get_buffer_size()
        buf_total_count = self._amp_replay_buffer.get_total_count()
        if (buf_total_count > buf_size):
            keep_probs = to_torch(np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]), device=self.ppo_device)
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            amp_obs = amp_obs[keep_mask]

        self._amp_replay_buffer.store({'amp_obs': amp_obs})
        return

    def _record_train_batch_info(self, batch_dict, train_info):
        train_info['disc_rewards'] = batch_dict['disc_rewards']
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)

        self.writer.add_scalar('losses/disc_loss', torch_ext.mean_list(train_info['disc_loss']).item(), frame)

        self.writer.add_scalar('info/disc_agent_acc', torch_ext.mean_list(train_info['disc_agent_acc']).item(), frame)
        self.writer.add_scalar('info/disc_demo_acc', torch_ext.mean_list(train_info['disc_demo_acc']).item(), frame)
        self.writer.add_scalar('info/disc_agent_logit', torch_ext.mean_list(train_info['disc_agent_logit']).item(), frame)
        self.writer.add_scalar('info/disc_demo_logit', torch_ext.mean_list(train_info['disc_demo_logit']).item(), frame)
        self.writer.add_scalar('info/disc_grad_penalty', torch_ext.mean_list(train_info['disc_grad_penalty']).item(), frame)
        self.writer.add_scalar('info/disc_logit_loss', torch_ext.mean_list(train_info['disc_logit_loss']).item(), frame)
        self.writer.add_scalar('info/disc_loss_least_sq', torch_ext.mean_list(train_info['disc_loss_least_sq']).item(), frame)
        self.writer.add_scalar('info/grad_disc_obs', torch_ext.mean_list(train_info['grad_disc_obs']).item(), frame)

        disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
        self.writer.add_scalar('info/disc_reward_mean', disc_reward_mean.item(), frame)
        self.writer.add_scalar('info/disc_reward_std', disc_reward_std.item(), frame)

        # Record disc experiment
        if self.disc_experiment:
            if self.pause_policy_updates and self.disc_expt_reset_disc:
                if not hasattr(self, 'writer_global_iter'):
                    self.writer_global_iter = 0

                for idx, val in enumerate(train_info['grad_disc_obs']):
                    self.writer.add_scalar('disc_experiment/grad_disc_obs/iter', val.item(), self.writer_global_iter+idx)

                for idx, val in enumerate(train_info['disc_loss_least_sq']):
                    self.writer.add_scalar('disc_experiment/disc_loss_least_sq/iter', val.item(), self.writer_global_iter+idx)

                for idx, val in enumerate(train_info['disc_demo_acc']):
                    self.writer.add_scalar('disc_experiment/disc_demo_acc/iter', val.item(), self.writer_global_iter+idx)

                for idx, val in enumerate(train_info['disc_agent_acc']):
                    self.writer.add_scalar('disc_experiment/disc_agent_acc/iter', val.item(), self.writer_global_iter+idx)

                for idx, val in enumerate(train_info['disc_combined_acc']):
                    self.writer.add_scalar('disc_experiment/disc_combined_acc/iter', val.item(), self.writer_global_iter+idx)

                self.writer_global_iter += len(train_info['grad_disc_obs'])


                
        return

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward)
        return

    ## New Addition ##
    # Evaluate Performance #
    def compute_performance_metrics(self, frame):
        """Compute performance metrics
        """
        self.set_eval()
        agent_pose_trajectories, agent_root_trajectories = self.run_policy()
        if not isinstance(agent_pose_trajectories, list):
            agent_pose_trajectories = [agent_pose_trajectories]
            agent_root_trajectories = [agent_root_trajectories]

        num_joints = agent_pose_trajectories[0].shape[-1]/3
        dt = self.vec_env.env.dt

        # Compute pose error
        avg_dtw_pose_error = 0
        dtw_start_time = time.time()
        for agent_pose_trajectory in agent_pose_trajectories:
            dtw_pose_error = 0
            for demo_traj in self.demo_trajectories:
                dtw_pose_error += ts_dtw(demo_traj.clone().cpu(), agent_pose_trajectory.clone().cpu()) / num_joints
            dtw_pose_error = dtw_pose_error/len(self.demo_trajectories)
            avg_dtw_pose_error += dtw_pose_error
        
        dtw_end_time = time.time()
        dtw_computation_performance = dtw_end_time - dtw_start_time
        avg_dtw_pose_error = avg_dtw_pose_error/len(agent_pose_trajectories)
        self.writer.add_scalar('mean_dtw_pose_error/step', avg_dtw_pose_error, frame)
        self.writer.add_scalar('dtw_computation_performance/step', dtw_computation_performance, frame)
                
        # Compute root body statistics
        avg_vel_norm = 0
        avg_acc_norm = 0
        avg_jerk_norm = 0
        mean_avg_sparc = 0
        for agent_root_trajectory in agent_root_trajectories:
            agent_root_velocity = get_series_derivative(agent_root_trajectory, dt)
            agent_root_acceleration = get_series_derivative(agent_root_velocity, dt)
            agent_root_jerk = get_series_derivative(agent_root_acceleration, dt)

            # Compute spectral arc length
            agent_velocity_profile = agent_root_velocity.cpu().numpy()
            mean_sparc = 0
            for ax in range(agent_velocity_profile.shape[-1]):
                ax_sparc_spectral_arc_len, _, _ = sparc(agent_velocity_profile[:,ax].squeeze(), 1/dt)
                mean_sparc += ax_sparc_spectral_arc_len
            mean_sparc = mean_sparc/agent_velocity_profile.shape[-1]
            mean_avg_sparc += mean_sparc

            mean_vel_norm = torch.mean(torch.linalg.norm(agent_root_velocity, dim=1))
            mean_acc_norm = torch.mean(torch.linalg.norm(agent_root_acceleration, dim=1))
            mean_jerk_norm = torch.mean(torch.linalg.norm(agent_root_jerk, dim=1))

            avg_vel_norm += mean_vel_norm
            avg_acc_norm += mean_acc_norm
            avg_jerk_norm += mean_jerk_norm
        
        avg_vel_norm = avg_vel_norm/len(agent_root_trajectories)
        avg_acc_norm = avg_acc_norm/len(agent_root_trajectories)
        avg_jerk_norm = avg_jerk_norm/len(agent_root_trajectories)
        mean_avg_sparc = mean_avg_sparc/len(agent_root_trajectories)

        self.writer.add_scalar('root_body_velocity/step', avg_vel_norm, frame)
        self.writer.add_scalar('root_body_acceleration/step', avg_acc_norm, frame)
        self.writer.add_scalar('root_body_jerk/step', avg_jerk_norm, frame)
        self.writer.add_scalar('spectral_arc_length/step', mean_avg_sparc, frame)

        # Compute demonstration data root body statistics
        if not hasattr(self, 'demo_root_body_stats'):
            # Compute root body statistics
            avg_vel_norm_demo = 0
            avg_acc_norm_demo = 0
            avg_jerk_norm_demo = 0
            mean_avg_sparc_demo = 0
            for demo_root_trajectory in self.demo_root_trajectories:
                demo_root_velocity = get_series_derivative(demo_root_trajectory, dt)
                demo_root_acceleration = get_series_derivative(demo_root_velocity, dt)
                demo_root_jerk = get_series_derivative(demo_root_acceleration, dt)

                # Compute spectral arc length
                demo_velocity_profile = demo_root_velocity.cpu().numpy()
                mean_sparc_demo = 0
                for ax in range(demo_velocity_profile.shape[-1]):
                    ax_sparc_spectral_arc_len_demo, _, _ = sparc(demo_velocity_profile[:,ax].squeeze(), 1/dt)
                    mean_sparc_demo += ax_sparc_spectral_arc_len_demo
                mean_sparc_demo = mean_sparc_demo/demo_velocity_profile.shape[-1]
                mean_avg_sparc_demo += mean_sparc_demo

                mean_vel_norm_demo = torch.mean(torch.linalg.norm(demo_root_velocity, dim=1))
                mean_acc_norm_demo = torch.mean(torch.linalg.norm(demo_root_acceleration, dim=1))
                mean_jerk_norm_demo = torch.mean(torch.linalg.norm(demo_root_jerk, dim=1))

                avg_vel_norm_demo += mean_vel_norm_demo
                avg_acc_norm_demo += mean_acc_norm_demo
                avg_jerk_norm_demo += mean_jerk_norm_demo
            
            avg_vel_norm_demo = avg_vel_norm_demo/len(self.demo_root_trajectories)
            avg_acc_norm_demo = avg_acc_norm_demo/len(self.demo_root_trajectories)
            avg_jerk_norm_demo = avg_jerk_norm_demo/len(self.demo_root_trajectories)
            mean_avg_sparc_demo = mean_avg_sparc_demo/len(self.demo_root_trajectories)
            self.demo_root_body_stats = {'spectral_arc_len_demo': mean_avg_sparc_demo, 'avg_vel_norm_demo': avg_vel_norm_demo.item(), 'avg_acc_norm_demo': avg_acc_norm_demo.item(), 'avg_jerk_norm_demo': avg_jerk_norm_demo.item()}
            self.writer.add_text('demo_root_body_stats', str(self.demo_root_body_stats), 0)

        # Print Performance Stats
        print("-----")
        print(f"Evaluating current policy's performance. Mean dynamic time warped pose error {avg_dtw_pose_error}. Time taken {dtw_computation_performance}")
        print(self.demo_root_body_stats)
        print({'spectral_arc_len_agent': mean_avg_sparc, 'avg_vel_norm_agent': avg_vel_norm.item(), 'avg_acc_norm_agent': avg_acc_norm.item(), 'avg_jerk_norm_agent':avg_jerk_norm.item()})
        print("-----")

        self.set_train()


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


    def compute_disc_performance(self):
        """Method used to evaluate the discriminator's predictions. 
        
        Returns the discriminator predictions and rewards for an episode
        """
        print("Evaluating discriminator predictions and rewards")
        is_deterministic = True
        max_steps = self._max_episode_length
        total_env_learnt_rewards = None
        total_disc_pred = None
        self.disc_eval_obses = self._env_reset_all()

        for n in range(4 * self.horizon_length):
            self.disc_eval_obses, _ = self._env_reset_done()

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.disc_eval_obses, masks)
            else:
                res_dict = self.get_action_values(self.disc_eval_obses)

            if is_deterministic:
                self.disc_eval_obses, rewards, dones, infos = self.env_step(res_dict['mus'])
            else:
                self.disc_eval_obses, rewards, dones, infos = self.env_step(res_dict['actions'])
                

            all_done_indices = dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]
            done_count = len(env_done_indices)

            # Compute preds and set done env preds to 0.0
            env_learnt_rewards = self._calc_disc_rewards(infos['amp_obs'])
            disc_pred = self._eval_disc(infos['amp_obs'])

            if total_env_learnt_rewards is not None:
                total_env_learnt_rewards += env_learnt_rewards.clone()
                total_disc_pred += disc_pred.clone()
            else:
                total_env_learnt_rewards = env_learnt_rewards.clone() 
                total_disc_pred = disc_pred.clone()
                
        mean_disc_reward = total_env_learnt_rewards.squeeze().mean()
        std_disc_reward = total_env_learnt_rewards.squeeze().std()
        mean_disc_pred = total_disc_pred.squeeze().mean()
        std_disc_pred = total_disc_pred.squeeze().std()

        self.disc_expt_mean_rew.append(mean_disc_reward.item())
        self.disc_expt_std_rew.append(std_disc_reward.item())
        self.disc_expt_mean_pred.append(mean_disc_pred.item())
        self.disc_expt_std_pred.append(std_disc_pred.item())

        self.obs = self._env_reset_all()


    def plot_disc_expt(self, iters_per_epoch=1):
        """ Plot the discriminator experiment results
        """
        import matplotlib.pyplot as plt
        import os
        import pickle
        epochs = iters_per_epoch * np.arange(len(self.disc_expt_mean_rew))

        disc_expt_mean_rew = np.array(self.disc_expt_mean_rew)
        disc_expt_std_rew = np.array(self.disc_expt_std_rew)
        disc_expt_mean_pred = np.array(self.disc_expt_mean_pred)
        disc_expt_std_pred = np.array(self.disc_expt_std_pred)

        saved_results = {'disc_expt_mean_rew':disc_expt_mean_rew, 'disc_expt_std_rew':disc_expt_std_rew, 
        'disc_expt_mean_pred':disc_expt_mean_pred, 'disc_expt_std_pred':disc_expt_std_pred}

        save_path = os.path.join(self.network_path, 
            self.config['name'] + f'policy_training_{self.disc_expt_policy_training}.pkl')

        with open(save_path, 'wb') as handle:
            pickle.dump(saved_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        std_interval = 1.0
        min_rew = disc_expt_mean_rew - std_interval*disc_expt_std_rew
        max_rew = disc_expt_mean_rew + std_interval*disc_expt_std_rew

        min_pred = disc_expt_mean_pred - std_interval*disc_expt_std_pred
        max_pred = disc_expt_mean_pred + std_interval*disc_expt_std_pred

        plt.plot(epochs, disc_expt_mean_rew, linewidth=1)
        plt.fill_between(epochs, min_rew, max_rew, alpha=0.2)
        plt.title(f'Discriminator behaviour on pausing policy updates at {self.disc_expt_policy_training}')
        plt.xlabel('Iters after pausing policy updates')
        plt.ylabel('Discriminator reward')
        plt.show()

        plt.plot(epochs, disc_expt_mean_pred, linewidth=1)
        plt.fill_between(epochs, min_pred, max_pred, alpha=0.2)
        plt.title(f'Discriminator behaviour on pausing policy updates at {self.disc_expt_policy_training}')
        plt.xlabel('Iters after pausing policy updates')
        plt.ylabel('Discriminator prediction')
        plt.show()




        

