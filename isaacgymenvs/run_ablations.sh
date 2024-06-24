#!/bin/bash

ncsn_cfg=$(python ./cfg/ablation_generator.py --model=ncsn)

# Run ncsn if the cfg is not empty or done
if [ "${ncsn_cfg}" = "done" ]; then
  echo "cmds done!"
  echo "------------"
else
  # python train_ncsn.py ${ncsn_cfg}
  echo ${ncsn_cfg}
fi


rl_cfg=$(python ./cfg/ablation_generator.py --model=rl)

# Run rl (either AMP or NEAR) if not done
if [ "${rl_cfg}" = "done" ]; then
  echo "cmds done!"
  echo "------------"
else
# python train.py ${rl_cfg}
  echo ${rl_cfg}
fi