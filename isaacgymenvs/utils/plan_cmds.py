import pandas as pd
import argparse
import os, sys

FILE_PATH = os.path.dirname(__file__)
sys.path.append(FILE_PATH)
sys.path.append(os.path.join(FILE_PATH, "../cfg/"))

from ablation_generator import generate_train_commands as generate_ablation_commands
from experiment_generator import generate_train_commands as generate_experiment_commands

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-rt", "--run_type", type=str, required=True, help="type of the experiment run (ablation or main experiments)")
    parser.add_argument("-n", "--num_runs", type=int, required=True, help="number of runs to execute")
    args = parser.parse_args()

    if args.run_type == "ablation":
        generate_ablation_commands()
        cmds = pd.read_pickle(os.path.join(FILE_PATH, "../cfg/ablation_cmds.pkl"))
    elif args.run_type == "experiment":
        generate_experiment_commands()
        cmds = pd.read_pickle(os.path.join(FILE_PATH, "../cgf/experiment_cmds.pkl"))

    
    non_assigned = cmds[cmds['job_assigned']==False]

    if non_assigned.empty:
        print(None)
    else:
        non_assigned_indices = list(non_assigned.index)
        if len(non_assigned_indices) >= args.num_runs:
            print(' '.join(map(str, non_assigned_indices[:args.num_runs])))
        else:
            print(' '.join(map(str, non_assigned_indices)))
            

