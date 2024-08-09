
## View motions that are in .npy format

import os
import json

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

# source fbx file path
fbx_file = "data/sfu_wushu_kicks.npy"

# import fbx file - make sure to provide a valid joint name for root_joint
motion = SkeletonMotion.from_file(fbx_file)
# visualize motion
plot_skeleton_motion_interactive(motion)


# view tpose
# print("Viewing the AMP T-pose")
# state = SkeletonState.from_file("data/amp_humanoid_tpose.npy")
# plot_skeleton_state(state)

# print("Viewing the CMU T-Pose (note: this is as per the horizontal dataset)")
# state = SkeletonState.from_file("data/horizontal_cmu_tpose.npy")
# plot_skeleton_state(state)