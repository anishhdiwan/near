# Copyright (c) 2021-2023, NVIDIA Corporation
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
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE..

from enum import Enum
import numpy as np
import torch
import os

from gym import spaces

from isaacgym import gymapi
from isaacgym import gymtorch

from isaacgymenvs.tasks.amp.humanoid_amp_base import HumanoidAMPBase, dof_to_obs, compute_humanoid_reward
from isaacgymenvs.tasks.amp.utils_amp import gym_util
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, calc_heading_quat_inv, quat_to_tan_norm, my_quat_rotate


NUM_AMP_OBS_PER_STEP = 13 + 52 + 28 + 12 # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]


class HumanoidAMP(HumanoidAMPBase):

    class StateInit(Enum):
        # Initialise at the default pose of the humanoid body
        Default = 0
        # Initialise at the start pose of the motion
        Start = 1
        # Initialise at a random pose in the motion
        Random = 2
        # Hybrid between Default and Random
        Hybrid = 3
        # Initialise environments at a uniform linspace in the motions
        Uniform = 4
        # Initialised at a random pose but the sampling is guided by a weighting. Meaning that some phases can be sampled more frequently. Allows for better exploration
        WeightedRandom = 5

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidAMP.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        motion_file = cfg['env'].get('motion_file', "amp_humanoid_backflip.npy")
        self.motion_style = os.path.splitext(motion_file)[0]

        # First try to find motions in the main assets folder. Then try in the dataset directory
        motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets/amp/motions', motion_file)
        if os.path.exists(motion_file_path):
            pass
        else:
            motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../custom_envs/data/humanoid', motion_file)
            assert os.path.exists(motion_file_path), "Provided motion file can not be found in the assets/amp/motions or data/humanoid directories"

        # motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/amp/motions/" + motion_file)
        self._load_motion(motion_file_path)

        # Infer the type of task reward based on the motion file
        self._infer_task_reward_type(motion_file)

        self.num_amp_obs = self._num_amp_obs_steps * NUM_AMP_OBS_PER_STEP

        self._amp_obs_space = spaces.Box(np.ones(self.num_amp_obs) * -np.Inf, np.ones(self.num_amp_obs) * np.Inf)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None
        self._reset_uniform_motion_ids = None

        return

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        return

    def get_num_amp_obs(self):
        return self.num_amp_obs

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)

    def fetch_amp_obs_demo(self, num_samples):
        dt = self.dt
        motion_ids = self._motion_lib.sample_motions(num_samples)

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
            
        motion_times0 = self._motion_lib.sample_time(motion_ids)
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps])
        motion_times = np.expand_dims(motion_times0, axis=-1)
        time_steps = -dt * np.arange(0, self._num_amp_obs_steps)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos,
                                      self._local_root_obs)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)

        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())
        return amp_obs_demo_flat

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)
        return

    def _load_motion(self, motion_file):
        self._motion_lib = MotionLib(motion_file=motion_file, 
                                     num_dofs=self.num_dof,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device,
                                     randomise_heading=self.env_assets)
        return

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._init_amp_obs(env_ids)
        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidAMP.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Start
              or self._state_init == HumanoidAMP.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Uniform):
            self._reset_uniform_state_init(env_ids)
        elif (self._state_init == HumanoidAMP.StateInit.WeightedRandom):
            self._reset_weighted_ref_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

        return
    
    def _reset_default(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self._reset_default_env_ids = env_ids
        return

    def _reset_uniform_state_init(self, env_ids):
        """Create a repeatable uniformly random init state for testing

        Samples chunks of motions ids [1,3,5,6 | 1,3,5,6 |...]
        Then samples chunks of init times [0. 0. 0. 0. | 0.1 0.1 0.1 0.1 | ...]

        Then resets the envs as per those phases in those motions
        """
        print("Resetting with uniform state initialisation")
        if self._reset_uniform_motion_ids is None:
            num_envs = env_ids.shape[0]
            num_motions = self._motion_lib.num_motions()
            num_init_states = 8
            chunk_size = int(num_envs/num_init_states)

            motion_id_elements = np.arange(num_motions)
            repetitions = int(chunk_size // len(motion_id_elements)) + 1
            repeated_motion_id_elements = np.tile(motion_id_elements, repetitions)
            np.random.shuffle(repeated_motion_id_elements)
            motion_ids = repeated_motion_id_elements[:chunk_size]
            motion_ids = np.tile(motion_ids, num_init_states)        
            
            phase_vals = np.linspace(start=0.1, stop=0.5, num=num_init_states)
            phase = np.repeat(phase_vals, chunk_size)

            motion_len = self._motion_lib._motion_lengths[motion_ids]
            motion_times = phase * motion_len

            self._reset_uniform_motion_ids = motion_ids
            self._reset_uniform_motion_times = motion_times

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(self._reset_uniform_motion_ids, self._reset_uniform_motion_times)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = self._reset_uniform_motion_ids.copy()
        self._reset_ref_motion_times = self._reset_uniform_motion_times.copy()
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        if (self._state_init == HumanoidAMP.StateInit.Random
            or self._state_init == HumanoidAMP.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == HumanoidAMP.StateInit.Start):
            motion_times = np.zeros(num_envs)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_weighted_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        motion_times = self._motion_lib.sample_time(motion_ids, beta_dist=True)

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps - 1])
        motion_times = np.expand_dims(motion_times, axis=-1)
        time_steps = -dt * (np.arange(0, self._num_amp_obs_steps - 1) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos,
                                      self._local_root_obs)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        if self.env_assets:
            humanoid_env_ids = self.num_actors_per_env * env_ids
        else:
            humanoid_env_ids = env_ids

        self._root_states[env_ids, 0:3] = root_pos
        self._root_states[env_ids, 3:7] = root_rot
        self._root_states[env_ids, 7:10] = root_vel
        self._root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        humanoid_env_ids_int32 = humanoid_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._all_root_states), 
                                                    gymtorch.unwrap_tensor(humanoid_env_ids_int32), len(humanoid_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                    gymtorch.unwrap_tensor(humanoid_env_ids_int32), len(humanoid_env_ids_int32))

        # Reset assets
        if self.env_assets:
            actor_root_pos = self.get_additional_actor_reset_poses(list(self.additional_actor_handles.keys()), self.num_actors_per_env-1, len(humanoid_env_ids), root_pos)
            actor_root_rot = torch.zeros((self.num_actors_per_env-1)*len(humanoid_env_ids), 4)
            actor_root_rot[:,-1] = 1.0
            actor_ids = torch.cat([humanoid_env_ids+i for i in range(1,self.num_actors_per_env)])
            self._all_root_states[actor_ids, 0:3] = actor_root_pos
            self._all_root_states[actor_ids, 3:7] = actor_root_rot.to('cuda:0', dtype=torch.float)
            self._all_root_states[actor_ids, 7:10] = torch.zeros_like(actor_root_pos, dtype=torch.float, device='cuda:0')
            self._all_root_states[actor_ids, 10:13] = torch.zeros_like(actor_root_pos, dtype=torch.float, device='cuda:0')

            actor_ids_int32 = actor_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._all_root_states), 
                                                        gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))

            self.reset_additional_actor_state(list(self.additional_actor_handles.keys()), env_ids)
        
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return

    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if (env_ids is None):
            self._curr_amp_obs_buf[:] = build_amp_observations(self._root_states, self._dof_pos, self._dof_vel, key_body_pos,
                                                                self._local_root_obs)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._root_states[env_ids], self._dof_pos[env_ids], 
                                                                    self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                    self._local_root_obs)
        return


    ## New Addition ##
    ## Does not affect AMP ##
    def fetch_demo_dataset(self):
        """Fetch all motions as trajectories of joints
        """

        # Arrays of motion features
        dt = self.dt
        motion_ids = np.array(range(self._motion_lib.num_motions()))
        motion_lengths = self._motion_lib._motion_lengths[motion_ids]
        num_frames = self._motion_lib._motion_num_frames[motion_ids]
        num_joints = self._motion_lib._get_num_bodies()

        # List of arrays of frame indices from the motion dataset such that the frames are approximately separated by a time gap of dt
        frame_idxs = []
        for i in range(len(motion_lengths)):
            phase = np.arange(0, motion_lengths[i], dt) / motion_lengths[i]
            phase = np.clip(phase, 0.0, 1.0)
            frame_idx = (phase * (num_frames[i] - 1)).astype(int)
            frame_idxs.append(frame_idx)

        # Sample joint poses at the selected frames for each motion
        joint_pose_trajectories = []
        parent_idx = None
        parent_joint_name = None
        for motion_id in motion_ids:
            frames = frame_idxs[motion_id]
            joint_pose_traj = np.empty([len(frames), num_joints, 3])
            curr_motion = self._motion_lib._motions[motion_id]
            joint_pose_traj = curr_motion.global_translation[frames]
            joint_pose_trajectories.append(joint_pose_traj)

            # Joint with parent -1 is the root
            if motion_id == 0:
                parent_idx = curr_motion.skeleton_tree.parent_indices.tolist().index(-1)
                parent_joint_name = curr_motion.skeleton_tree.node_names[parent_idx]

        return joint_pose_trajectories, [parent_idx, parent_joint_name]


    def reset_all(self):
        """Reset all envs
        Returns:
            Observation dictionary, indices of environments being reset
        """
        env_ids = torch.arange(self.num_envs, device=self.reset_buf.device, dtype=self.reset_buf.dtype)
        self.reset_idx(env_ids)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict

    
    def _compute_reward(self, actions):
        """Inherit to add reward types
        """
        if self.env_assets:
            goal_type = list(self.additional_actor_handles.keys())[0]
            if goal_type == "flagpole":
                self.rew_buf[:] = compute_humanoid_target_reaching_reward(self._rigid_body_pos, self._rigid_body_vel, self._additional_actor_rigid_body_pos, self.body_ids_dict['pelvis'])
            elif goal_type == "football":
                raise NotImplementedError
            elif goal_type == "box":
                self.rew_buf[:] = compute_humanoid_target_punching_reward(self._rigid_body_pos, self._rigid_body_vel, self._additional_actor_rigid_body_pos, self.additional_actor_state, self.body_ids_dict)


        else:
            reward_buffers = []

            for reward_type in self.reward_types:
                if reward_type == "time":
                    reward_buffers.append(compute_humanoid_reward(self.obs_buf))
                
                if reward_type == "dist":
                    try:
                        reward_buffers.append(compute_humanoid_dist_reward(self._rigid_body_pos, self._prev_dist_rew_body_pos, self.body_ids_dict['pelvis']))
                    except AttributeError:
                        # Small positive reward for first timestep to avoid nan errors
                        reward_buffers.append(torch.full_like(self.obs_buf[:, 0], 0.001))

                    self._prev_dist_rew_body_pos = self._rigid_body_pos.clone()

                if reward_type == "height":
                    reward_buffers.append(compute_humanoid_height_reward(self._rigid_body_pos, self.body_ids_dict['pelvis']))

            # Add all types of rewards proportionally
            if len(reward_buffers) > 1:
                reward_buffers = [reward_buffer/len(reward_buffers) for reward_buffer in reward_buffers]
                self.rew_buf[:] = torch.add(*reward_buffers)
            else:
                self.rew_buf[:] = reward_buffers[0]

        return

    def _infer_task_reward_type(self, motion_file):
        """Infer the type of task reward given the motion file
        """

        task_reward_dict = {
            "amp_humanoid_walk": ["time", "dist"],
            "amp_humanoid_run": ["time", "dist"],
            "amp_humanoid_crane_pose": ["time"],
            "amp_humanoid_cartwheel": ["time", "dist"],
            "amp_humanoid_jump_in_place": ["time", "height"],
            "amp_humanoid_martial_arts_bassai": ["time"],
            "amp_humanoid_boxing": ["time"],
            "amp_humanoid_right_punch": ["time"],
            "amp_humanoid_left_punch": ["time"],
            "amp_humanoid_indian_dance_bharatnatyam1": ["time"],
            "amp_humanoid_indian_dance_bharatnatyam2": ["time"],
            "amp_humanoid_indian_dance_kuchipudi": ["time"],
            "amp_humanoid_single_left_punch": ["time"],
            "amp_humanoid_overhead_kick": ["time"],
            "amp_humanoid_spin_kick": ["time", "height"],
            "amp_humanoid_crescent_kick": ["time"],
            "amp_humanoid_marching": ["time", "dist"],
            "amp_humanoid_bow": ["time"],
            "amp_humanoid_zombie_walk": ["time", "dist"],
        }

        motion_file = os.path.splitext(motion_file)[0]

        try:
            self.reward_types = task_reward_dict[motion_file]
        except KeyError:
            self.reward_types = ["time"]

    def get_goal_features(self):
        goal_type = list(self.additional_actor_handles.keys())[0]
        if goal_type == "flagpole":
            agent_rigid_body_pos = self._rigid_body_pos.clone()
            target_pos = self._additional_actor_rigid_body_pos.clone()
            root_body_id = self.body_ids_dict['pelvis']
            
            agent_root_pos = agent_rigid_body_pos[:, root_body_id, :]
            agent_root_pos[:,-1] = 0.0
            target_pos[:,-1] = 0.0

            relative_target_pos = target_pos - agent_root_pos
            return relative_target_pos[:,:-1]

        elif goal_type == "box":
            agent_rigid_body_pos = self._rigid_body_pos.clone()
            target_pos = self._additional_actor_rigid_body_pos.clone()
            root_body_id = self.body_ids_dict['pelvis']
            
            agent_root_pos = agent_rigid_body_pos[:, root_body_id, :]
            agent_root_pos[:,-1] = 0.0
            target_pos[:,-1] = 0.0

            relative_target_pos = target_pos - agent_root_pos
            additional_actor_state = self.additional_actor_state.float()

            return torch.cat([relative_target_pos[:,:-1],additional_actor_state.unsqueeze(1)], dim=-1)

        elif goal_type == "football":
            raise NotImplementedError



#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def build_amp_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos)

    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs

@torch.jit.script
def compute_humanoid_dist_reward(rigid_body_pos, prev_rigid_body_pose, root_body_id):
    # type: (Tensor, Tensor, int) -> Tensor

    # Cap the displacement in timestep dt to something reasonable
    max_disp = 0.5
    root_body_x_pos = rigid_body_pos.clone()[:, root_body_id, 0]
    # No need to clone as its already cloned
    prev_root_body_x_pos = prev_rigid_body_pose[:, root_body_id, 0]
    disp = root_body_x_pos - prev_root_body_x_pos
    disp = torch.clamp(disp, min=0.0, max=max_disp)

    # Scale to [0,1]
    reward = disp/max_disp

    return reward


@torch.jit.script
def compute_humanoid_height_reward(rigid_body_pos, root_body_id):
    # type: (Tensor, int) -> Tensor

    # Reward up to 2.5m of vertical root position
    max_disp = 2.5
    root_body_z_pos = rigid_body_pos.clone()[:, root_body_id, 2]
    root_body_z_pos = torch.clamp(root_body_z_pos, min=0.0, max=max_disp)

    # Scale to [0,1]
    reward = root_body_z_pos/max_disp

    return reward


@torch.jit.script
def compute_humanoid_target_reaching_reward(agent_rigid_body_pos, agent_rigid_body_vel, target_pos, root_body_id):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor

    agent_root_pos = agent_rigid_body_pos.clone()[:, root_body_id, :]
    agent_root_pos[:,-1] = 0.0
    target_pos = target_pos.clone()
    target_pos[:,-1] = 0.0
    agent_root_body_vel = agent_rigid_body_vel.clone()[:, root_body_id, :]
    agent_root_body_vel[:,-1] = 0.0
    # agent_root_body_vel = agent_root_body_vel.norm(p=2, dim=1)

    pos_error = target_pos - agent_root_pos
    pos_error_norm = pos_error.norm(p=2, dim=1)
    root_to_target_unit_vector = (pos_error.T/pos_error_norm).T
    # agent_vel_in_unit_vector_direction = torch.sum(agent_root_body_vel * root_to_target_unit_vector, dim=1)
    normalised_agent_vel_in_unit_vector_direction = torch.sum(agent_root_body_vel * root_to_target_unit_vector, dim=1)/agent_root_body_vel.norm(p=2, dim=1)
    # heading_error = 1.0 - agent_vel_in_unit_vector_direction
    heading_error = normalised_agent_vel_in_unit_vector_direction

    desired_velocity = 2.0
    # Minimise squared positional error, minimise heading error, minimise squared errror of velocity norm with some desired velocity.
    reward = 0.6*torch.exp(-0.5*pos_error_norm**2) + 0.3*(1 - 2/(1+torch.exp(5*heading_error))) + 0.1*(1 - (agent_root_body_vel.norm(p=2, dim=1) - desired_velocity)**2) # 0.3*torch.exp(-(torch.maximum(torch.zeros_like(heading_error), heading_error))**2)
    reward[pos_error_norm <= 1.00] += 0.5

    return reward


@torch.jit.script
def compute_humanoid_target_punching_reward(agent_rigid_body_pos, agent_rigid_body_vel, target_pos, target_punched, body_ids_dict):
    # type: (Tensor, Tensor, Tensor, Tensor, Dict[str, int]) -> Tensor

    root_body_id = body_ids_dict["pelvis"]
    punching_body_id = body_ids_dict["left_hand"]
    agent_root_pos = agent_rigid_body_pos.clone()[:, root_body_id, :]
    agent_root_pos[:,-1] = 0.0
    agent_punching_body_pos = agent_rigid_body_pos.clone()[:, punching_body_id, :]
    agent_punch_height = agent_punching_body_pos[:,-1].clone()
    agent_punching_body_pos[:,-1] = 0.0
    target_pos = target_pos.clone()
    target_pos[:,-1] = 0.0
    agent_root_body_vel = agent_rigid_body_vel.clone()[:, root_body_id, :]
    agent_root_body_vel[:,-1] = 0.0
    agent_punching_body_vel = agent_rigid_body_vel.clone()[:, punching_body_id, :]
    agent_punching_body_vel[:,-1] = 0.0
    # agent_root_body_vel = agent_root_body_vel.norm(p=2, dim=1)

    pos_error = target_pos - agent_root_pos
    pos_error_norm = pos_error.norm(p=2, dim=1)
    root_to_target_unit_vector = (pos_error.T/pos_error_norm).T
    normalised_agent_vel_in_unit_vector_direction = torch.sum(agent_root_body_vel * root_to_target_unit_vector, dim=1)/agent_root_body_vel.norm(p=2, dim=1)
    # heading_error = 1.0 - normalised_agent_vel_in_unit_vector_direction
    heading_error = normalised_agent_vel_in_unit_vector_direction


    punching_body_vel_in_unit_vector_direction = torch.sum(agent_punching_body_vel * root_to_target_unit_vector, dim=1)
    punching_body_pos_error = target_pos - agent_punching_body_pos
    punching_body_pos_error_norm = punching_body_pos_error.norm(p=2, dim=1)

    desired_velocity = 2.0
    desired_punch_height = 1.2
    
    reward_near = 0.3*torch.exp(-2*punching_body_pos_error_norm**2) + 0.4*torch.clamp(0.667*punching_body_vel_in_unit_vector_direction ,0,1) + 0.3*(1 - (agent_punch_height - desired_punch_height)**2)
    reward_far = 0.6*torch.exp(-0.5*pos_error_norm**2) + 0.3*(1 - 2/(1+torch.exp(5*heading_error))) + 0.1*(1 - (agent_root_body_vel.norm(p=2, dim=1) - desired_velocity)**2) #0.3*torch.exp(-heading_error) # 0.3*torch.exp(-(torch.maximum(torch.zeros_like(heading_error), heading_error))**2))
    reward_far[target_punched] = 1.0
    reward_near[target_punched] = 1.0
    near_mask = (~target_punched) & (pos_error_norm < 1.5)
    # reward = torch.where(near_mask, reward_near, reward_far)
    # reward_far[near_mask] += reward_near[near_mask]
    reward_far[near_mask] = reward_near[near_mask]

    return reward_far




