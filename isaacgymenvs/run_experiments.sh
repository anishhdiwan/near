#!/bin/bash

datetime=$(date '+%d-%m-%y-%H-%M-%S')
cfg=$(python ./cfg/experiment_generator.py)
cmd="${cfg}_${datetime}"

python train.py $cmd