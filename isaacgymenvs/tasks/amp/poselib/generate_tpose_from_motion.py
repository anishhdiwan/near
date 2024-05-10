import os
import sys
import json
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

from pathlib import Path
home = str(Path.home())
# fbx data path
data_path = home + "/thesis_background/Datasets/CMU_humanoid_fbx/"
data_index = pd.read_csv(data_path + "cmu_mocap_task_index.csv")
task_name = ['walk']
task_index = data_index.loc[data_index['task'].isin(task_name)]
task_index['motion_file'] = data_path + "CMU_fbx/" + task_index['motion_index'] + ".fbx"
motion_files = task_index['motion_file'].to_list()



"""
This script imports an FBX format motion file from the given dataset depending on the given task. It then extracts a SkletonState from 
the first frame and exports this into a T-pose. Can be used further for retargeting motions to other T-poses.
"""

fbx_file = motion_files[0]
print(f"Loading {fbx_file}")
# import fbx file - make sure to provide a valid joint name for root_joint
SkeletonMotion.tpose_from_fbx(
    fbx_file_path=fbx_file,
    root_joint="Hips",
    fps=60,
    file_name="horizontal_cmu_tpose"
)