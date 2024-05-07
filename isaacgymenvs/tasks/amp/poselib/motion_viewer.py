
## View motions that are in .npy format

import os
import json

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

# source fbx file path
fbx_file = "data/01_01_cmu.npy"

# import fbx file - make sure to provide a valid joint name for root_joint
motion = SkeletonMotion.from_file(fbx_file)

# visualize motion
plot_skeleton_motion_interactive(motion)
