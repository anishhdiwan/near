#!/bin/bash

ncsn_output=$(python ~/thesis_background/IsaacGymEnvs/isaacgymenvs/cfg/ablation_generator.py --model=ncsn)
# ncsn_cfg=$(echo "$ncsn_output" | jq -r '.cmd')
# job_idx=$(echo "$ncsn_output" | jq -r '.job_idx')
ncsn_cfg=$(python ~/thesis_background/IsaacGymEnvs/isaacgymenvs/utils/json_parser.py "${ncsn_output}" '.cmd')
job_idx=$(python ~/thesis_background/IsaacGymEnvs/isaacgymenvs/utils/json_parser.py "${ncsn_output}" '.job_idx')


echo "Job assigned a row ID ${job_idx}"
echo "-----------"

# Run ncsn if the cfg is not empty or done
if [ "${ncsn_cfg}" = "done" ]; then
  echo "cmds done!"
  echo "------------"
else
  echo ${ncsn_cfg}
  # python ~/thesis_background/IsaacGymEnvs/isaacgymenvs/train_ncsn.py ${ncsn_cfg}
fi

sleep 1.0

rl_cfg=$(python ~/thesis_background/IsaacGymEnvs/isaacgymenvs/cfg/ablation_generator.py --model=rl --job_idx=${job_idx})

# Run rl (either AMP or NEAR) if not done
if [ "${rl_cfg}" = "done" ]; then
  echo "cmds done!"
  echo "------------"
else
  echo ${rl_cfg}
  # python ~/thesis_background/IsaacGymEnvs/isaacgymenvs/train.py ${rl_cfg}
  
fi