"""
This script generates k random seeds to seed the k trials of each experiment. Seeds are stored in a .yaml file in the same directory
"""

import random
import yaml
from datetime import datetime

start = 0
end = int(5e4)
k = 5

seeds = [{"seeds": random.sample(range(start, end), k)}]

with open(f'experiment_seeds_{datetime.now().strftime("_%d-%H-%M-%S")}.yml', 'w') as yaml_file:
    yaml.dump(seeds, yaml_file, default_flow_style=False)