

"""
This script takes in a dataset of motions in .fbx format, then tranforms them to .npy.

This is done based on some given task and an the motion index
"""

import os
import sys
import pandas as pd
import argparse
pd.options.mode.chained_assignment = None  # default='warn'

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from pathlib import Path


if __name__ == "__main__":

    print("""

    This script converts .fbx format motion files to .npy format files. It also has an option to just visualise a motions. Run with the option --help for more info.
    Available options for the task argument can be found in the "task" column of the cmu-mocap-index sheet (https://docs.google.com/spreadsheets/d/1v8lSJoWWd4lB4HvEyHF8n48rNzNNzfmHYWz_yVBDTfc/edit?usp=sharing)
    
    Note: The passed path must have the "cmu_mocap_task_index.csv" file and the CMU_fbx directory containing motion files. Motion files can be obtained from https://academictorrents.com/details/8e21416d1584981ef3e9d8a97ee4278f93390623
    
    """)

    parser = argparse.ArgumentParser() 

    parser.add_argument("-t", "--task", type=str, default="walk", required=False, help="name of the task to convert/view")
    parser.add_argument("-ns", "--no_save", default=True, action='store_false', required=False, help="when not to save the converted .npy files")
    parser.add_argument("-pv", "--preview", default=False, action='store_true', required=False, help="preview the first few motion files")
    parser.add_argument("-v", "--view", default=False, action='store_true', required=False, help="view all motion files")
    parser.add_argument("-spth", "--save_path", type=str, default="/thesis_background/Datasets/CMU_humanoid_fbx/", required=False, help="path to save the data (relative to home)")
    parser.add_argument("-m", "--motion", type=str, default="", required=False, help="motion index if visualising a specific motion file")
    args = parser.parse_args()

    save = args.no_save
    view = args.view
    preview = args.preview
    
    if preview == True:
        if view == True or save == True:
            print("Warning! Cannot view all motions or save motions if preview is set to True. Proceeding to preview")
            view = False
            save = False
    else:
        if view == True and save == True:
            print("Warning! Cannot view all motions and save motions at the same time. Proceeding to save")
            view = False

    if save == False and view == False and preview == False:
        print("Warning! No feasible options passed. Previewing motions")
        preview = True


    home = str(Path.home())
    # fbx data path
    data_path = home + args.save_path

    if args.motion == "":
        data_index = pd.read_csv(data_path + "cmu_mocap_task_index.csv")
        task_name = [args.task.lower()]
        task_index = data_index.loc[data_index['task'].str.lower().isin(task_name)]
        task_index['motion_file'] = data_path + "CMU_fbx/" + task_index['motion_index'] + ".fbx"
        motion_files = task_index['motion_file'].to_list()
        motion_indices = task_index['motion_index'].to_list()

        savepath = data_path + "cmu_" + task_name[0].replace(" ", "_") + "_task/" 
        if save:
            if not os.path.exists(savepath):
                os.makedirs(savepath)

        print(f"Number of mo-cap files for the {task_name[0]} task: {len(motion_files)}")

        if preview:
            motion_files = motion_files[:1]

    else:
        save = False
        preview = True
        motion_files = []
        motion_pth = data_path + "CMU_fbx/" + args.motion + ".fbx"
        motion_files.append(motion_pth)


    for idx, fbx_file in enumerate(motion_files):
        print(f"Loading {fbx_file}")
        # import fbx file - make sure to provide a valid joint name for root_joint
        motion = SkeletonMotion.from_fbx(
            fbx_file_path=fbx_file,
            root_joint="Hips",
            fps=60,
        )

        # save motion in npy format
        if save:
            file_name = savepath + "cmu_" + motion_indices[idx] + ".npy"
            motion.to_file(file_name)

        # visualize motion
        if view or preview:
            plot_skeleton_motion_interactive(motion)

        print("-----------------")