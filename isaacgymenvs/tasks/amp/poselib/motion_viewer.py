
## View motions that are in .npy format

import os
import json

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

# source npy file path
# npy_file = "/home/anishdiwan/thesis_background/IsaacGymEnvs/isaacgymenvs/custom_envs/data/humanoid/amp_cmu_cartwheel_sideways_task/amp_cmu_90_02.npy"
npy_file = "data/amp_humanoid_cartwheel.npy"
# npy_file = "/home/anishdiwan/thesis_background/Datasets/CMU_humanoid_fbx/cmu_roll_task/cmu_90_34.npy"

# import fbx file - make sure to provide a valid joint name for root_joint
motion = SkeletonMotion.from_file(npy_file)
# motion.tensor = motion.tensor[:slide_idx]
# visualize motion

# motion.to_file("some path")
plot_skeleton_motion_interactive(motion)




# view tpose
# print("Viewing the AMP T-pose")
# state = SkeletonState.from_file("data/amp_humanoid_tpose.npy")
# plot_skeleton_state(state)

# print("Viewing the CMU T-Pose (note: this is as per the horizontal dataset)")
# state = SkeletonState.from_file("data/horizontal_cmu_tpose.npy")
# plot_skeleton_state(state)