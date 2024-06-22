import matplotlib.pyplot as plt
import numpy as np
import os, sys
from experiment_plotter import Colours
import argparse
from pathlib import Path
import yaml
import re

FILE_PATH = os.path.join(os.path.dirname(__file__))
sys.path.append(FILE_PATH)


def generate_play_commands(algo, checkpoints, trials):
    play_cmds = Path(os.path.join(FILE_PATH, f"{algo}_play_cmds.yaml"))
    if play_cmds.is_file():
        # Avoid overwriting automatically
        pass

    else:
        pending_cmds = []
        for checkpoint in checkpoints:
            for trial in trials:
                if algo == "AMP":
                    checkpoints_pth = os.path.join(FILE_PATH, f"../runs/{trial}/nn")
                    trial_checkpoints = [f for f in os.listdir(checkpoints_pth) if os.path.isfile(os.path.join(checkpoints_pth, f))]

                    # Define the regex pattern to match the known checkpoint
                    pattern = rf'_{checkpoint}\.pth$'
                    for filename in trial_checkpoints:
                        if re.search(pattern, filename):
                            pending_cmds.append(f"train.py task=Humanoid{algo} test=True checkpoint=runs/{trial}/nn/{filename}")

                elif algo=="DMP":
                    pending_cmds.append(f"train_ncsn.py task=Humanoid{algo} test=True \
++train.params.config.dmp_config.inference.eb_model_checkpoint=ncsn_runs/{trial}/nn/checkpoint.pth \
++train.params.config.dmp_config.inference.running_mean_std_checkpoint=ncsn_runs/{trial}/nn/running_mean_std.pth")


        cmds = [{"pending_cmds":pending_cmds}]
        with open(os.path.join(FILE_PATH, f"{algo}_play_cmds.yaml"), 'w') as yaml_file:
            yaml.dump(cmds, yaml_file, default_flow_style=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algo", type=str, required=True, help="whether to plot AMP rewards or DMP rewards")
    parser.add_argument('-t', '--trials', nargs='+', required=True, help='the experiment trials to plot')
    parser.add_argument('-c', '--checkpoints', nargs='+', required=True, help='the checkpoints to plot')
    args = parser.parse_args()

    generate_play_commands(args.algo, args.checkpoints, args.trials)


    # open the training commands yaml file
    with open(os.path.join(FILE_PATH, f"{args.algo}_play_cmds.yaml")) as stream:
        try:
            cmds = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    try:
        # pop the first one
        command_to_pass = cmds[0]['pending_cmds'].pop(0)
    except IndexError as e:
        print("done")
        quit()
    
    # save the training commands yaml file
    with open(os.path.join(FILE_PATH, f"{args.algo}_play_cmds.yaml"), 'w') as yaml_file:
        yaml.dump(cmds, yaml_file, default_flow_style=False) 

    print(command_to_pass)







