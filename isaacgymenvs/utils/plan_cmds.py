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
        cmds_path = os.path.join(FILE_PATH, "../cfg/ablation_cmds.pkl")
        cmds = pd.read_pickle(cmds_path)
    elif args.run_type == "experiment":
        generate_experiment_commands()
        cmds_path = os.path.join(FILE_PATH, "../cfg/train_cmds.pkl")
        cmds = pd.read_pickle(cmds_path)

    
    non_assigned = cmds[cmds['job_assigned']==False]

    if non_assigned.empty:
        print(None)
    else:
        non_assigned_indices = list(non_assigned.index)
        if len(non_assigned_indices) >= args.num_runs:
            cmds.loc[non_assigned_indices[:args.num_runs], "job_assigned"] = True
            cmds.loc[non_assigned_indices[:args.num_runs], "ncsn_cmd_passed"] = True
            cmds.loc[non_assigned_indices[:args.num_runs], "rl_cmd_passed"] = True
            cmds.to_pickle(cmds_path)
            print(' '.join(map(str, non_assigned_indices[:args.num_runs])))
        else:
            cmds.loc[non_assigned_indices, "job_assigned"] = True
            cmds.loc[non_assigned_indices, "ncsn_cmd_passed"] = True
            cmds.loc[non_assigned_indices, "rl_cmd_passed"] = True
            cmds.to_pickle(cmds_path)
            print(' '.join(map(str, non_assigned_indices)))
            

