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


import numpy as np
import os
import torch
import random

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, get_axis_params, calc_heading_quat_inv, \
     exp_map_to_quat, quat_to_tan_norm, my_quat_rotate, calc_heading_quat_inv

from ..base.vec_task import VecTask

DOF_BODY_IDS = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
NUM_OBS = 13 + 52 + 28 + 12 # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
NUM_ACTIONS = 28

NUM_FEATURES = {
"dof_names":['abdomen', 'neck', 'right_shoulder', 'right_elbow', 'left_shoulder', 'left_elbow', 
'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle'],
"num_features":[3, 3, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3]
}

UPPER_BODY_MASK = ['neck', 'right_shoulder', 'right_elbow', 'left_shoulder', 'left_elbow']
LOWER_BODY_MASK = ['abdomen', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle']

# Dict with actor name as key and a dict of actor params as value

ADDITIONAL_ACTORS = {
    "football": {"file": "amp/soccerball.urdf", "num_instances":1, "texture":"amp/meshes/soccer_ball.png"},
    "flagpole": {"file": "amp/flagpole.urdf", "num_instances":1, "colour": gymapi.Vec3(204/255, 24/255, 0/255)}, 
    "box": {"file": "amp/box.urdf", "num_instances":1, "colour": gymapi.Vec3(0.8, 0.8, 0.8)} #"texture":"amp/meshes/cardboard.png"
    }


DEMO_CHAR_COLOUR = gymapi.Vec3(141/255, 182/255, 0/255)
LEARNT_CHAR_COLOURS = [gymapi.Vec3(10/255, 235/255, 255/255), gymapi.Vec3(0.4706, 0.549, 0.6863)]
RANDOMISE_COLOURS = True
TOP_VIEW = False
ASSET_VIEW = False
PAN_CAMERA = False
PAN_SPEED = -0.035
AGENT_COLOUR = LEARNT_CHAR_COLOURS[0]

KEY_BODY_NAMES = ["right_hand", "left_hand", "right_foot", "left_foot"]
POSSIBLE_BODY_NAMES = ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand',
'left_upper_arm', 'left_lower_arm', 'left_hand', 'right_thigh', 'right_shin', 'right_foot', 'left_thigh', 'left_shin', 'left_foot']

class HumanoidAMPBase(VecTask):

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = config

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.randomize = self.cfg["task"]["randomize"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()


        # Whether the scene has other physical actors (here, an actor could also be a non-agent asset like a football) 
        self.env_assets = self.cfg["env"].get("envAssets", [])
        if self.env_assets == []:
            self.env_assets = False
        else:
            global ADDITIONAL_ACTORS
            ADDITIONAL_ACTORS = dict((k, ADDITIONAL_ACTORS[k]) for k in self.env_assets if k in ADDITIONAL_ACTORS)
            self.env_assets = True
            if "box" in ADDITIONAL_ACTORS.keys():
                self._contact_bodies.append("left_hand")
    
        self.num_actors_per_env = 1
        if self.env_assets:
            self.num_actors_per_env = 1 + sum([ADDITIONAL_ACTORS[additional_actor]["num_instances"] for additional_actor in list(ADDITIONAL_ACTORS.keys())])

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._all_root_states = gymtorch.wrap_tensor(actor_root_state)
        self._root_states = self._all_root_states.view(self.num_envs, self.num_actors_per_env, 13)[:, :1, :].squeeze(dim=1)
        self._initial_root_states = self._root_states.clone()
        self._initial_root_states[:, 7:13] = 0

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        right_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "right_shoulder_x")
        left_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "left_shoulder_x")
        self._initial_dof_pos[:, right_shoulder_x_handle] = 0.5 * np.pi
        self._initial_dof_pos[:, left_shoulder_x_handle] = -0.5 * np.pi

        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[...,:self.humanoid_num_bodies, 0:3]
        self._rigid_body_rot = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[...,:self.humanoid_num_bodies, 3:7]
        self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[...,:self.humanoid_num_bodies, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[...,:self.humanoid_num_bodies, 10:13]
        self.all_contact_forces = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = self.all_contact_forces.view(self.num_envs, self.num_bodies, 3)[:,:self.humanoid_num_bodies,:]

        self._additional_actor_rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[...,self.humanoid_num_bodies:self.num_bodies, 0:3].squeeze(dim=1)
        self._additional_actor_contact_forces = self.all_contact_forces.view(self.num_envs, self.num_bodies, 3)[...,self.humanoid_num_bodies:self.num_bodies,:].squeeze(dim=1)

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        if self.viewer != None:
            self._init_camera()
            
        return

    def get_obs_size(self):
        return NUM_OBS

    def get_action_size(self):
        return NUM_ACTIONS

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        return

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        return

    def set_char_color(self, col):
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            handle = self.humanoid_handles[i]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
        asset_file = "mjcf/amp_humanoid.xml"

        if "asset" in self.cfg["env"]:
            #asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.humanoid_num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.humanoid_num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.humanoid_num_joints = self.gym.get_asset_joint_count(humanoid_asset)
        self.num_bodies = self.humanoid_num_bodies
        self.num_dof = self.humanoid_num_dof
        self.num_joints = self.humanoid_num_joints

        print(f"Loading humanoid asset. Asset has {self.num_bodies} bodies {self.num_dof} dofs and {self.num_joints} joints")

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.89, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        if self.env_assets:
            additional_actor_assets = {}
            additional_actor_start_poses = {}
            self.additional_actor_handles = {}
            self.additional_actor_visual = {}
            for additional_actor_name, additional_actor_params in ADDITIONAL_ACTORS.items():
                # Define additional actor start poses
                additional_actor_start_poses[additional_actor_name] = self.get_additional_actor_start_poses(additional_actor_name, additional_actor_params["num_instances"])
                
                # Load additional actor assets
                additional_actor_asset_options = self.get_additional_actor_asset_options(additional_actor_name)
                additional_actor_asset = self.gym.load_asset(self.sim, asset_root, additional_actor_params["file"], additional_actor_asset_options)
                additional_actor_assets[additional_actor_name] = additional_actor_asset

                actor_num_bodies = self.gym.get_asset_rigid_body_count(additional_actor_asset)
                actor_num_dof = self.gym.get_asset_dof_count(additional_actor_asset)
                actor_num_joints = self.gym.get_asset_joint_count(additional_actor_asset)
                print(f"Loading {additional_actor_name} asset. Asset has {actor_num_bodies} bodies {actor_num_dof} dofs and {actor_num_joints} joints")

                self.num_bodies += actor_num_bodies * additional_actor_params["num_instances"]
                self.num_dof += actor_num_dof * additional_actor_params["num_instances"]
                self.num_joints += actor_num_joints * additional_actor_params["num_instances"]

                # Define the list of additional actor handles
                self.additional_actor_handles[additional_actor_name] = []

                # Define asset visual properties
                if "texture" in list(additional_actor_params.keys()):
                    self.additional_actor_visual[additional_actor_name] = {"texture": self.gym.create_texture_from_file(self.sim, additional_actor_params["texture"])}
                if "colour" in list(additional_actor_params.keys()):
                    self.additional_actor_visual[additional_actor_name] = {"colour": additional_actor_params["colour"]}
            
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            contact_filter = 0
            
            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, contact_filter, 0)
            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            if RANDOMISE_COLOURS:
                colour = gymapi.Vec3(*np.random.uniform(0.0, 1.0, 3))
            else:
                colour = AGENT_COLOUR
            for j in range(self.humanoid_num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, colour)

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

            if self.env_assets:
                for additional_actor_name in list(additional_actor_assets.keys()):
                    for ind in range(ADDITIONAL_ACTORS[additional_actor_name]["num_instances"]):
                        additional_actor_handle = self.gym.create_actor(env_ptr, additional_actor_assets[additional_actor_name], additional_actor_start_poses[additional_actor_name][ind], f"{additional_actor_name}_{ind}", i, contact_filter, 0)
                        self.additional_actor_handles[additional_actor_name].append(additional_actor_handle)
                        if "texture" in list(ADDITIONAL_ACTORS[additional_actor_name].keys()):
                            self.gym.set_rigid_body_texture(env_ptr, additional_actor_handle, 0, gymapi.MESH_VISUAL, self.additional_actor_visual[additional_actor_name]["texture"])
                        elif "colour" in list(ADDITIONAL_ACTORS[additional_actor_name].keys()):
                            if RANDOMISE_COLOURS:
                                asset_colour = colour
                            else:
                                asset_colour = self.additional_actor_visual[additional_actor_name]["colour"]
                            self.gym.set_rigid_body_color(env_ptr, additional_actor_handle, 0, gymapi.MESH_VISUAL, asset_colour)

            if (self._pd_control):
                dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
                dof_prop["driveMode"] = gymapi.DOF_MODE_POS
                self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.humanoid_num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, handle)
        self._build_dof_ids_dict(env_ptr, handle)
        self._build_body_ids_dict(env_ptr, handle)
        self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, handle)
        
        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return

    def _build_pd_action_offset_scale(self):
        num_joints = len(DOF_OFFSETS) - 1
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = DOF_OFFSETS[j]
            dof_size = DOF_OFFSETS[j + 1] - DOF_OFFSETS[j]

            if (dof_size == 3):
                lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)
                
                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)
        return

    def _compute_reset(self):
        if self.env_assets:
            goal_type = list(self.additional_actor_handles.keys())[0]
            if goal_type == "flagpole":
                self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_target_reaching_reset(self.reset_buf, self.progress_buf,
                                                        self._contact_forces, self._contact_body_ids,
                                                        self._rigid_body_pos, self._additional_actor_rigid_body_pos, self.max_episode_length,
                                                        self._enable_early_termination, self._termination_height, self.body_ids_dict['pelvis'], self.motion_style)
            elif goal_type == "football":
                raise NotImplementedError
            elif goal_type == "box":
                self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_target_punching_reset(self.reset_buf, self.progress_buf,
                                                        self._contact_forces, self._contact_body_ids,
                                                        self._rigid_body_pos, self._additional_actor_rigid_body_pos, self.additional_actor_state, self.max_episode_length,
                                                        self._enable_early_termination, self._termination_height, self.body_ids_dict['pelvis'], self.motion_style)

        else:
            self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                    self._contact_forces, self._contact_body_ids,
                                                    self._rigid_body_pos, self.max_episode_length,
                                                    self._enable_early_termination, self._termination_height)


        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        if self.env_assets:
            self.refresh_additional_actor_state()
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._root_states
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        else:
            root_states = self._root_states[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
        
        obs = compute_humanoid_observations(root_states, dof_pos, dof_vel,
                                            key_body_pos, self._local_root_obs)
        return obs

    def _reset_actors(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()

        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return

    def post_physics_step(self):
        self.progress_buf += 1

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()
        
        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self, mode='rgb_array'):
        if self.viewer and self.camera_follow:
            self._update_camera()

        super().render(mode)
        return

    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_body_ids_dict(self, env_ptr, actor_handle):
        body_ids = {}
        for body_name in POSSIBLE_BODY_NAMES:
            try:
                body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
                body_ids[body_name] = body_id
            except Exception:
                pass

        self.body_ids_dict = body_ids

    def _build_dof_ids_dict(self, env_ptr, actor_handle):
        dof_ids = {}
        for dof_name in self.gym.get_actor_dof_names(env_ptr, actor_handle):
            try:
                dof_id = self.gym.find_actor_dof_handle(env_ptr, actor_handle, dof_name)
                dof_ids[dof_name] = dof_id
            except Exception:
                pass

        self.dof_ids_dict = dof_ids

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._root_states[0, 0:3].cpu().numpy()

        if TOP_VIEW:
            cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], 
                                self._cam_prev_char_pos[1] - 6.0, 
                                4.0)
        else:
            cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], 
                                  self._cam_prev_char_pos[1] - 4.0, 
                                  2.0)


        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)

        if ASSET_VIEW:
            char_root_pos = self._all_root_states.view(self.num_envs, self.num_actors_per_env, 13)[:, 1, :].squeeze(dim=1)[0, 0:3].cpu().numpy()
            char_root_pos[0] -= 4.5
        else:
            char_root_pos = self._root_states[0, 0:3].cpu().numpy()
        
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])

        if PAN_CAMERA:
            if self.control_steps < 25:
                pass
            else:
                cam_pos += np.array([-PAN_SPEED*(cam_pos[1]-char_root_pos[1])/4, PAN_SPEED*(cam_pos[0]-char_root_pos[0])/4, 0.0])

        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
                                  char_root_pos[1] + cam_delta[1], 
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return


    def get_additional_actor_asset_options(self, additional_actor_name):
        """
        Get asset options for the actor
        """

        if additional_actor_name == "football":
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.max_angular_velocity = 20.0
            asset_options.max_linear_velocity = 10.0
            asset_options.linear_damping = 0.01
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            # asset_options.use_mesh_materials = True
            return asset_options

        elif additional_actor_name == "flagpole":
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            return asset_options

        elif additional_actor_name == "box":
            asset_options = gymapi.AssetOptions()
            # asset_options.fix_base_link = True
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            return asset_options



    def get_additional_actor_start_poses(self, additional_actor_name, num_instances):
        """ Return a list of start poses for the actors based on some initialisation logic. Also set up any states for the actors
        """

        if additional_actor_name == "football":
            start_poses = []
            # Initialise the asset at some random point away from the agent with z = 0.15
            x_pos = np.linspace(1.0, 2.0, num_instances)
            y_pos = np.linspace(1.0, 2.0, num_instances)
            for i in range(num_instances):
                start_p = np.array([x_pos[i], y_pos[i], 0.15])
                actor_start_pose = gymapi.Transform()
                actor_start_pose.p = gymapi.Vec3(*list(start_p))
                actor_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                start_poses.append(actor_start_pose)
            
            return start_poses


        elif additional_actor_name == "flagpole":
            assert num_instances == 1, "There can only be one flagpole asset. Change the code in humanoid_amp_base to change this"

            # Initialise the asset at some random point away from the agent with z = 0
            start_p = np.random.uniform(low=2.0, high=3.0, size=3)
            start_p[-1] = 0.0
            actor_start_pose = gymapi.Transform()
            actor_start_pose.p = gymapi.Vec3(*list(start_p))
            actor_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            return [actor_start_pose]


        elif additional_actor_name == "box":
            assert num_instances == 1, "There can only be one box asset. Change the code in humanoid_amp_base to change this"

            # Initialise the asset at some random point away from the agent with z = 0
            start_p = np.random.uniform(low=2.0, high=2.0, size=3)
            start_p[-1] = 1.001
            actor_start_pose = gymapi.Transform()
            actor_start_pose.p = gymapi.Vec3(*list(start_p))
            actor_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            self.additional_actor_state = torch.full((self.num_envs*(self.num_actors_per_env-1),), False).to('cuda:0')

            return [actor_start_pose]

    def reset_additional_actor_state(self, additional_actor_name, additional_actor_ids):
        if additional_actor_name[0] in ["football", "flagpole"]:
            pass
        elif additional_actor_name[0] == "box":
            self.additional_actor_state[additional_actor_ids] = False
    
    def refresh_additional_actor_state(self):
        if list(self.additional_actor_handles.keys())[0] in ["football", "flagpole"]:
            pass
        elif list(self.additional_actor_handles.keys())[0] == "box":
            # The box has been punched if the punching hand and the box both have non zero forces and if the root body is not below the reset threshold
            if not hasattr(self, "non_punching_bodies"):
                non_punching_bodies = [self.body_ids_dict[body_name] for body_name in POSSIBLE_BODY_NAMES if body_name not in ["left_hand"]]
                self.non_punching_bodies = to_torch(sorted(non_punching_bodies), device=self.device, dtype=torch.long) 
            
            # List of envs where the punching body has non zero contact forces 
            masked_contact_buf = self._contact_forces.clone()
            masked_contact_buf[:, self.non_punching_bodies, :] = 0
            punching_body_contact = torch.any(masked_contact_buf > 0.5, dim=-1)
            punching_body_contact = torch.any(punching_body_contact, dim=-1)

            # List of envs where the box has non zero contact forces
            masked_additional_actor_contact_buf = self._additional_actor_contact_forces.clone()
            masked_additional_actor_contact_buf[:, -1] = 0 # Ignore z axis forces to discount gravity induced normal force
            box_contact = torch.any(masked_additional_actor_contact_buf>0.5, dim=-1)

            # List of envs where the body height is above a threshold
            # root_body_height = self._rigid_body_pos[:, self.body_ids_dict["pelvis"], 2]
            # not_fallen = root_body_height > self._termination_height
            # not_fallen = torch.any(not_fallen, dim=-1)

            # Considered to have fallen if the hand is below some threshold
            punching_body_height = self._rigid_body_pos[:, self.body_ids_dict["left_hand"], 2]
            not_fallen = punching_body_height > 0.5
            not_fallen = torch.any(not_fallen, dim=-1)

            has_punched = torch.logical_and(torch.logical_and(punching_body_contact, box_contact), not_fallen)
            self.additional_actor_state[:] = has_punched.to('cuda:0')

    def get_goal_completion(self):
        if list(self.additional_actor_handles.keys())[0] in ["football", "flagpole"]:
            root_body_id = self.body_ids_dict["pelvis"]
            agent_root_pos = self._rigid_body_pos.clone()[:, root_body_id, :]
            agent_root_pos[:,-1] = 0.0
            target_pos = self._additional_actor_rigid_body_pos.clone()
            target_pos[:,-1] = 0.0
            pos_error = target_pos - agent_root_pos
            pos_error_norm = pos_error.norm(p=2, dim=1)
            return pos_error_norm <= 1.00
        elif list(self.additional_actor_handles.keys())[0] == "box":
            return self.additional_actor_state
        


    def get_additional_actor_reset_poses(self, additional_actor_name, num_instances, num_env_ids, agent_pos, motion_style=None):
        """ Return a list of reset poses for the actors. Outputs a tensor of shape [num_env_ids*num_instances, 3]
        """
        theta_min = 120
        theta_max = 120

        if motion_style is not None:
            motion_style = os.path.splitext(motion_style)[0]

        if additional_actor_name[0] == "football":
            min_dist = -1.0
            max_dist = 1.0
        elif additional_actor_name[0] == "flagpole":
            if motion_style == "amp_humanoid_run":
                min_dist = 5.0
                max_dist = 10.0
            else:
                min_dist = 2.0
                max_dist = 6.0
        elif additional_actor_name[0] == "box":
            theta_min = theta_max = 30
            if motion_style == "amp_humanoid_run":
                min_dist = 6.0
                max_dist = 8.0
            elif motion_style in ["amp_humanoid_strike", "amp_humanoid_single_left_punch"]:
                min_dist = 1.0
                max_dist = 1.2
            else:
                if np.random.uniform() > 0.7:
                    min_dist = 1.0
                    max_dist = 1.2
                else:
                    min_dist = 1.2
                    max_dist = 4.0


        if additional_actor_name[0] == "football":
            # Initialise the asset at some random point away from the agent with z = 0.15
            reset_poses = torch.FloatTensor(num_instances*num_env_ids, 3).uniform_(min_dist, max_dist)   
            reset_poses[:,-1] = 0.15         
            return reset_poses

        elif additional_actor_name[0] == "box":
            assert num_instances == 1, "There can only be one box asset. Change the code in humanoid_amp_base to change this"

            # Initialise the asset in a band defined by a min and max radius relative to the agent with z = 0.0
            # directions = torch.randn(num_instances*num_env_ids, 2)
            # norms = torch.norm(directions, dim=1, keepdim=True) 
            # unit_vectors = directions/norms  # Normalize to get points on the unit sphere
            # radii = torch.empty(num_instances*num_env_ids).uniform_(min_dist, max_dist)
            # reset_poses = unit_vectors * radii[:, None]

            theta_min = torch.deg2rad(torch.tensor(theta_min).float())
            theta_max = torch.deg2rad(torch.tensor(theta_max).float())
            angles = torch.empty(num_instances*num_env_ids).uniform_(-theta_min, theta_max)
            radii = torch.empty(num_instances*num_env_ids).uniform_(min_dist, max_dist)
            x = radii * torch.cos(angles)
            y = radii * torch.sin(angles)
            reset_poses = torch.stack((x, y), dim=1)

            reset_poses = reset_poses.to('cuda:0', dtype=torch.float)
            reset_poses += agent_pos[:, :-1] # Translate reset pos relative to the agent
            reset_poses = torch.cat([reset_poses, torch.full((reset_poses.shape[0], 1), 1.001).to('cuda:0', dtype=torch.float)], dim=1) 

            return reset_poses

        elif additional_actor_name[0] == "flagpole":
            assert num_instances == 1, "There can only be one flagpole asset. Change the code in humanoid_amp_base to change this"

            theta_min = torch.deg2rad(torch.tensor(theta_min).float())
            theta_max = torch.deg2rad(torch.tensor(theta_max).float())
            angles = torch.empty(num_instances*num_env_ids).uniform_(-theta_min, theta_max)
            radii = torch.empty(num_instances*num_env_ids).uniform_(min_dist, max_dist)
            x = radii * torch.cos(angles)
            y = radii * torch.sin(angles)
            reset_poses = torch.stack((x, y), dim=1)
            reset_poses = reset_poses.to('cuda:0', dtype=torch.float)
            reset_poses += agent_pos[:, :-1] # Translate reset pos relative to the agent
            reset_poses = torch.cat([reset_poses, torch.zeros(reset_poses.shape[0], 1).to('cuda:0', dtype=torch.float)], dim=1) 

            # Initialise the asset in a band defined by a min and max radius relative to the agent with z = 0.0
            # directions = torch.randn(num_instances*num_env_ids, 2)
            # norms = torch.norm(directions, dim=1, keepdim=True) 
            # unit_vectors = directions/norms  # Normalize to get points on the unit sphere
            # radii = torch.empty(num_instances*num_env_ids).uniform_(min_dist, max_dist)
            # reset_poses = unit_vectors * radii[:, None]
            # reset_poses = reset_poses.to('cuda:0', dtype=torch.float)
            # reset_poses += agent_pos[:, :-1] # Translate reset pos relative to the agent
            # reset_poses = torch.cat([reset_poses, torch.zeros(reset_poses.shape[0], 1).to('cuda:0', dtype=torch.float)], dim=1) 

            return reset_poses


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_obs(pose):
    # type: (Tensor) -> Tensor
    #dof_obs_size = 64
    #dof_offsets = [0, 3, 6, 9, 12, 13, 16, 19, 20, 23, 24, 27, 30, 31, 34]
    dof_obs_size = 52
    dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset:(dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs

@torch.jit.script
def compute_humanoid_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
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
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_height):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_height
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated


@torch.jit.script
def compute_humanoid_target_reaching_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, target_pos,
                           max_episode_length, enable_early_termination, termination_height, root_body_id, motion_style):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float, int, str) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_height
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    agent_root_pos = rigid_body_pos.clone()[:, root_body_id, :]
    agent_root_pos[:,-1] = 0.0
    target_pos = target_pos.clone()
    target_pos[:,-1] = 0.0
    relative_target_pos = target_pos - agent_root_pos
    pos_error = torch.norm(relative_target_pos, p=2, dim=1)
    if motion_style == "humanoid_amp_run":
        min_pos_error_thresh = 1.25
        max_pos_error_thresh = 12.0
    else:
        min_pos_error_thresh = 1.25
        max_pos_error_thresh = 8.0

    reset = torch.where(((pos_error <= min_pos_error_thresh) | (pos_error >= max_pos_error_thresh)), torch.ones_like(reset_buf), reset)

    return reset, terminated


@torch.jit.script
def compute_humanoid_target_punching_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, target_pos, has_punched,
                           max_episode_length, enable_early_termination, termination_height, root_body_id, motion_style):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float, int, str) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_height
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    # Reset it agent too far away from box
    agent_root_pos = rigid_body_pos.clone()[:, root_body_id, :]
    agent_root_pos[:,-1] = 0.0
    target_pos = target_pos.clone()
    target_height = target_pos[:,-1].clone()
    target_pos[:,-1] = 0.0
    relative_target_pos = target_pos - agent_root_pos
    pos_error = torch.norm(relative_target_pos, p=2, dim=1)
    if motion_style == "humanoid_amp_run":
        max_pos_error_thresh = 15.0
    else:
        max_pos_error_thresh = 5.0

    reset = torch.where((pos_error >= max_pos_error_thresh), torch.ones_like(reset_buf), reset)

    # Reset if the target has been punched
    # has_punched = has_punched.clone()
    # reset = torch.where(has_punched, torch.ones_like(reset_buf), reset)

    # Reset if the target has fallen down (not necessarily from punching)
    target_fallen = target_height < 0.6
    reset = torch.where(target_fallen, torch.ones_like(reset_buf), reset)

    return reset, terminated