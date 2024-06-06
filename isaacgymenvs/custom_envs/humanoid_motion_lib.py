
import os
import time
import numpy as np
from typing import Dict, Any
from isaacgym import gymapi
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib
from isaacgymenvs.utils.torch_jit_utils import get_axis_params, to_torch
from isaacgymenvs.tasks.humanoid_amp import build_amp_observations
import torch

EXISTING_SIM = None
KEY_BODY_NAMES = ["right_hand", "left_hand", "right_foot", "left_foot"]
NUM_OBS_PER_STEP = 13 + 52 + 28 + 12 # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM


class HumanoidMotionDataset():

    def __init__(self, motion_file, humanoid_cfg, device, encode_temporal_feature=False):
        """Motion library for the humanoid motions dataset

        Args:
            motion_file: Humanoid motions file
            humanoid_cfg (DictConfig): A dictionary of config params for setting up the assets, envs, and sim in isaacgym
            device (torch device): Device to use 
        """
        self.device = device
        self.encode_temporal_feature = encode_temporal_feature

        # First try to find motions in the main assets folder. Then try in the dataset directory
        motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets/amp/motions', motion_file)
        if os.path.exists(motion_file_path):
            self.motion_file = motion_file_path
        else:
            motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/humanoid', motion_file)
            assert os.path.exists(motion_file_path), "Provided motion file can not be found in the assets/amp/motions or data/humanoid directories"
            self.motion_file = motion_file_path


        self.humanoid_cfg = humanoid_cfg
        self._num_obs_steps = self.humanoid_cfg["env"].get("numObsSteps", 2)
        self.num_obs = self._num_obs_steps * NUM_OBS_PER_STEP
        self._local_root_obs = self.humanoid_cfg["env"]["localRootObs"]

        self.control_freq_inv = self.humanoid_cfg["env"].get("controlFrequencyInv", 1)
        dt = self.humanoid_cfg["sim"].get("dt", 0.0166)
        self.dt = self.control_freq_inv * dt

        self.num_dof, self._key_body_ids = self._get_motionlib_args()

        self.batch_size = None
        self.sample_buffer = None
        self.buffer_size = None
        self.gotten_items = 0

        self._load_motions()


    def _get_motionlib_args(self):
        """Get the arguments needed to instantiate the motion sampling class from isaacgymenvs.

        Create a sim and fetch asset properties from gym (not passing in asset properties manually to avoid hard coding)
        TODO: find an alternative way to get properties!!
        """

        self.gym = gymapi.acquire_gym()

        sim_params = self.__parse_sim_params(self.humanoid_cfg["physics_engine"], self.humanoid_cfg["sim"])
        sim = _create_sim_once(self.gym)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/amp_humanoid.xml"

        if "asset" in self.humanoid_cfg["env"]:
            #asset_root = self.humanoid_cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file = self.humanoid_cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        humanoid_asset = self.gym.load_asset(sim, asset_root, asset_file, asset_options)
        num_dof = self.gym.get_asset_dof_count(humanoid_asset)

        spacing = self.humanoid_cfg.env.envSpacing
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(np.sqrt(self.humanoid_cfg.env.numEnvs))

        # Creating only one env (not in a loop)
        env_ptr = self.gym.create_env(sim, lower, upper, num_per_row)
        start_pose = gymapi.Transform()
        self.up_axis_idx = 2
        contact_filter = 0
        start_pose.p = gymapi.Vec3(*get_axis_params(0.89, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", 0, contact_filter, 0)

        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)

        print("Obtained motion details. Destroying sim")
        time.sleep(1.)
        self.gym.destroy_sim(sim)

        return num_dof, body_ids

    def _load_motions(self):
        self._motion_lib = MotionLib(motion_file=self.motion_file, 
                                     num_dofs=self.num_dof,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)


    def sample_paired_traj(self, num_samples, encode_temporal_feature=False):
        out_shape = (num_samples, self._num_obs_steps, NUM_OBS_PER_STEP)
        dt = self.dt
        motion_ids = self._motion_lib.sample_motions(num_samples)

        if encode_temporal_feature:
            motion_times0, motion_phase_1, motion_phase_0 = self._motion_lib.sample_time(motion_ids, return_phase=True, dt=dt)
        else:
            motion_times0 = self._motion_lib.sample_time(motion_ids)

        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_obs_steps])
        motion_times = np.expand_dims(motion_times0, axis=-1)
        

        time_steps = -dt * np.arange(0, self._num_obs_steps)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos,
                                      self._local_root_obs)
        obs_demo = obs_demo.view(out_shape)


        if encode_temporal_feature:
            # Add temporal information to state vectors
            motion_phase_1 = np.expand_dims(np.expand_dims(motion_phase_1, axis=-1), axis=-1)
            motion_phase_0 = np.expand_dims(np.expand_dims(motion_phase_0, axis=-1), axis=-1)
            motion_phase = torch.cat((to_torch(motion_phase_1, device=self.device), to_torch(motion_phase_0, device=self.device)), dim=1)
            obs_demo_temporal = torch.cat((motion_phase, obs_demo),-1)
            obs_demo_temporal = obs_demo_temporal.view(-1, self.num_obs+self._num_obs_steps)
        
            return obs_demo_temporal
        
        else:
            obs_demo_flat = obs_demo.view(-1, self.num_obs)
            return obs_demo_flat

    def set_batch_and_buffer_size(self, batch_size, buffer_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def __len__(self):
        # return self._motion_lib.get_total_length()
        # Since data is sampled from within a buffer (and a buffer is a trajectory of length buffer_size), the length of the dataset is just the buffer size.
        return self.buffer_size

    def __getitem__(self, idx):
        assert self.batch_size is not None, "Please set dataset batch size using set_batch_and_buffer_size() before creating the dataloader"
        assert self.buffer_size is not None, "Please set data buffer size using set_batch_and_buffer_size() before creating the dataloader"
        
        # Build a temporary buffer of traj samples
        if self.sample_buffer is None:
            self.sample_buffer = self._create_sample_buffer(self.buffer_size)
            
        # Sample an element from the buffer and update the current item count
        sample = self.sample_buffer[idx]
        self.gotten_items += 1

        # If a batch is already sampled then reset the buffer and item counter
        if self.gotten_items == self.buffer_size:
            self.sample_buffer = None
            self.gotten_items = 0

        return sample

    def _create_sample_buffer(self, buffer_size):
        return self.sample_paired_traj(buffer_size, encode_temporal_feature=self.encode_temporal_feature)


    def __parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Args:
            physics_engine: which physics engine to use. "physx" or "flex"
            config_sim: dict of sim configuration parameters
        Returns
            IsaacGym SimParams object with updated settings.
        """
        sim_params = gymapi.SimParams()

        # check correct up-axis
        if config_sim["up_axis"] not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        # assign general sim parameters
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim.get("pipeline", "gpu") == "gpu"
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        # configure physics parameters
        if physics_engine == "physx":
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        else:
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        # return the configured params
        return sim_params


class HumanoidMotionLib():

    def __init__(self, motion_file, humanoid_cfg, device, encode_temporal_feature=False):
        self.dataset = HumanoidMotionDataset(motion_file, humanoid_cfg, device, encode_temporal_feature=encode_temporal_feature)

    def get_dataloader(self, batch_size, buffer_size, shuffle=False):
        """Returns a dataloader that can be used to sample a batch of observation pairs. 

        """
        self.dataset.set_batch_and_buffer_size(batch_size, buffer_size)
        dataloader = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=batch_size,
                    num_workers=0,
                    shuffle=shuffle,
                    )

        return dataloader