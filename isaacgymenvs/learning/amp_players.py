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

import torch 
import numpy as np

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer
import matplotlib.pyplot as plt

import isaacgymenvs.learning.common_player as common_player

import copy


class AMPPlayerContinuous(common_player.CommonPlayer):

    def __init__(self, params):
        config = params['config']

        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._disc_reward_scale = config['disc_reward_scale']
        self._print_disc_prediction = config.get('print_disc_prediction', False)
        
        super().__init__(params)
        return

    def restore(self, fn):
        super().restore(fn)
        if self._normalize_amp_input:
            checkpoint = torch_ext.load_checkpoint(fn)
            self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        return
    
    def _build_net(self, config):
        super()._build_net(config)

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()
        return

    def _post_step(self, info):
        super()._post_step(info)
        if self._print_disc_prediction:
            self._amp_debug(info)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            try:
                config['amp_input_shape'] = self.env.amp_observation_space.shape
            except AttributeError as e:
                config['amp_input_shape'] = self.env.paired_observation_space.shape

        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']

        return config

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs.to(self.device))
            amp_rewards = self._calc_amp_rewards(amp_obs.to(self.device))
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward)
        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

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
            prob = 1.0 / (1.0 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r


    def visualise_discriminator_landscape(self):
        """Visualise the discriminator values vs distance from the data manifold 

        """
        device = self.device
        num_batches_to_sample = 15
        batch_size = 128
        demo_observations = self.env.fetch_amp_obs_demo(int(num_batches_to_sample*batch_size))

        self.plot_disc_curve(demo_observations)
        


    def plot_disc_curve(self, samples):
        """Plot a curve with the average disc value of a set of samples on the y-axis and the distance of the samples from the demo dataset on the x-axis
        """
        # Absolute values of the range [-r, r] of a uniform distribution from which demo data is perturbed
        demo_sample_max_distances = np.linspace(0, 10, 100)

        avg_disc_val = np.zeros_like(demo_sample_max_distances)
        avg_amp_rew = np.zeros_like(demo_sample_max_distances)

        for idx, max_dist in enumerate(demo_sample_max_distances):
            if max_dist == 0.0:
                perturbed_samples = copy.deepcopy(samples)
            else:
                perturbed_samples = copy.deepcopy(samples) + (max_dist -2*max_dist*torch.rand(samples.shape, device=samples.device))

            disc_pred = self._eval_disc(perturbed_samples.to(self.device))
            amp_rewards = self._calc_amp_rewards(perturbed_samples.to(self.device))['disc_rewards']

            avg_disc_val[idx] = disc_pred.squeeze().mean()
            avg_amp_rew[idx] = amp_rewards.squeeze().mean()

        
        plt.figure(figsize=(8, 6))
        plt.plot(demo_sample_max_distances, avg_disc_val, label="disc value")
        # plt.legend()
        plt.xlabel("max perturbation r (where sample = sample + unif[-r,r])")
        plt.ylabel("avg disc(sample)")
        plt.title(f"Avg disc value vs distance from demo data")
        plt.show()
        

        plt.figure(figsize=(8, 6))
        plt.plot(demo_sample_max_distances, avg_amp_rew, label="amp reward")
        # plt.legend()
        plt.xlabel("max perturbation r (where sample = sample + unif[-r,r])")
        plt.ylabel("avg amp rew")
        plt.title(f"Avg amp reward vs distance from demo data")
        plt.show()


    def visualise_2d_disc(self):
        """Visualise the discriminator function for 2D maze environment

        """
        device = self.device
        viz_min = 0
        viz_max = 512
        kernel_size = 3 # must be odd
        grid_steps = 128
        window_idx_left = int((kernel_size - 1)/2)
        window_idx_right = int((kernel_size + 1)/2)


        xs = torch.linspace(viz_min, viz_max, steps=grid_steps)
        ys = torch.linspace(viz_min, viz_max, steps=grid_steps)
        x, y = torch.meshgrid(xs, ys, indexing='xy')

        grid_points = torch.cat((x.flatten().view(-1, 1),y.flatten().view(-1,1)), 1).to(device=self.device)
        grid_points = grid_points.reshape(grid_steps,grid_steps,2)
        disc_grid = torch.zeros(grid_steps,grid_steps,1)
        rew_grid = torch.zeros(grid_steps,grid_steps,1)

        for i in range(grid_points.shape[0]):
            for j in range(grid_points.shape[1]):
                if i in [viz_min,viz_max] or j in [viz_min,viz_max]:
                    pass
                    
                else:
                    window = grid_points[i-window_idx_left:i+window_idx_right,j-window_idx_left:j+window_idx_right,:]
                    grid_pt_window = torch.zeros_like(window)
                    grid_pt_window[:,:,:] = grid_points[i,j]

                    obs_pairs = torch.cat((window, grid_pt_window), 2)
                    obs_pairs = obs_pairs.reshape(-1,4)

                    disc_pred = self._eval_disc(obs_pairs.to(self.device))
                    amp_rewards = self._calc_amp_rewards(obs_pairs.to(self.device))['disc_rewards']

                    mean_amp_rew = torch.mean(amp_rewards).item()
                    mean_disc_pred = torch.mean(disc_pred).item()
                    disc_grid[i,j] = mean_disc_pred
                    rew_grid[i,j] = mean_amp_rew

        disc_grid = disc_grid.reshape(-1,x.shape[0])
        rew_grid = rew_grid.reshape(-1,x.shape[0])


        plt.figure(figsize=(8, 6))
        mesh = plt.pcolormesh(x.cpu().cpu().detach().numpy(), y.cpu().detach().numpy(), disc_grid.cpu().detach().numpy(), cmap ='gray')
        plt.gca().invert_yaxis()
        plt.xlabel("env - x")
        plt.ylabel("env - y")
        plt.title(f"Maze Env disc(s,s') | Mean disc pred in agent's reachable set")
        plt.colorbar(mesh)
        plt.show()


        plt.figure(figsize=(8, 6))
        mesh = plt.pcolormesh(x.cpu().cpu().detach().numpy(), y.cpu().detach().numpy(), rew_grid.cpu().detach().numpy(), cmap ='gray')
        plt.gca().invert_yaxis()
        plt.xlabel("env - x")
        plt.ylabel("env - y")
        plt.title(f"Maze Env amp_reward(s,s') | Mean amp reward in agent's reachable set")
        plt.colorbar(mesh)
        plt.show()