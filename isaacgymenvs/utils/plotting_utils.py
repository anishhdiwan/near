import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation as scaledtf
import numpy as np
import csv
import pandas as pd

# Importing from the file path
import sys
import os
FILE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(FILE_PATH)

# https://sashamaps.net/docs/tools/20-colors/
Colours = [
	'#800000',  # Maroon (99.99%)
	'#4363d8',  # Blue (99.99%)
	'#ffe119',  # Yellow (99.99%)
	'#e6beff',  # Lavender (99.99%)
	'#f58231',  # Orange (99.99%)
	'#3cb44b',  # Green (99%)
	'#000075',  # Navy (99.99%)
	'#e6194b',  # Red (99%)
	'#46f0f0',  # Cyan (99%)
	'#f032e6',  # Magenta (99%)
	'#9a6324',  # Brown (99%)
	'#008080',  # Teal (99%)
	'#911eb4',  # Purple (95%*)
	'#aaffc3',  # Mint (99%)
	'#ffd8b1',  # Apiroct (95%)
	'#bcf60c',  # Lime (95%)
	'#fabed4',  # Pink (99%)
	'#808000',  # Olive (95%)
	'#fffac8',  # Beige (99%)
	#'#a9a9a9',
	#'#ffffff',
	#'#000000'
]

def search_logs(logdir):
    """Search for .csv log files in the passed logdir. Create separate pandas dataframes for each data tag

    Logs need to be stored as follows

    logdir
    |
    |-------- Experiment 1
            |
            |---- Summaries
                |
                |----tag1.csv    
                |----tag2.csv
    |
    |
    |-------- Experiment k

    Args:
        logdir (Str): Directory in which to search for logs

    Returns:
        logs (dict): A dictionary with keys as experiment names and values as sub-dicts. Each sub-dict has keys as tags and values as the path of the .csv file for that tag
    """

    tags = []
    logdir = os.path.join(FILE_PATH, logdir)
    # Get dirname : dirpath/summaries as a dictionary
    experiments = {name: os.path.join(logdir, name, "summaries") for name in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, name))}

    logs = {}
    for dirname, dirpath in experiments.items():
        # Get tag: filepath in an experiment directory
        # experiment_logs = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
        experiment_logs = {os.path.splitext(f.rsplit('-tag-')[1])[0]: os.path.join(logdir, dirname, "summaries", f) for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))}

        logs[dirname] = experiment_logs

    return logs


    def plot(tag):
        pass

    def multi_plot(tags):
        pass
        

if __name__ == "__main__":
    runs_path = "../ncsn_runs/"
    logs = search_logs(runs_path)

