

"""
This script takes in a dataset of motions in .fbx format, then tranforms them to .npy.

This is done based on some given task and an the motion index
"""


import os
import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

from pathlib import Path
home = str(Path.home())
# fbx data path
data_path = home + "/thesis_background/Datasets/CMU_humanoid_fbx/"


data_index = pd.read_csv(data_path + "cmu_mocap_task_index.csv")
task_name = ['forward jump']
task_index = data_index.loc[data_index['task'].isin(task_name)]
task_index['motion_file'] = data_path + "CMU_fbx/" + task_index['motion_index'] + ".fbx"
motion_files = task_index['motion_file'].to_list()
motion_indices = task_index['motion_index'].to_list()

savepath = data_path + "cmu_" + task_name[0] + "_task/" 
if not os.path.exists(savepath):
    os.makedirs(savepath)

print(f"Number of mo-cap files for the {task_name[0]} task: {len(motion_files)}")

for idx, fbx_file in enumerate(motion_files):
    print(f"Loading {fbx_file}")
    # import fbx file - make sure to provide a valid joint name for root_joint
    motion = SkeletonMotion.from_fbx(
        fbx_file_path=fbx_file,
        root_joint="Hips",
        fps=60,
    )

    # save motion in npy format
    file_name = savepath + "cmu_" + motion_indices[idx] + ".npy"
    motion.to_file(file_name)

    # visualize motion
    # plot_skeleton_motion_interactive(motion)

    print("-----------------")