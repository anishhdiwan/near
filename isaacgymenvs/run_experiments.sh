#!/bin/bash

ncsn_cfg=$(python ./cfg/experiment_generator.py --model=ncsn)

# Run ncsn if the cfg is not empty
if [ "${ncsn_cfg}" != "" ]
then
#   python train_ncsn.py ${ncsn_cfg}
  echo ${ncsn_cfg}
else
  echo "NCSN Skipped"
  echo "------------"
fi

# Run rl (either AMP or DMP)
rl_cfg=$(python ./cfg/experiment_generator.py --model=rl)
# python train.py ${rl_cfg}
echo ${rl_cfg}