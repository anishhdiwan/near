#!/bin/bash

job_idx=$(python ~/thesis_background/IsaacGymEnvs/isaacgymenvs/utils/plan_cmds.py --run_type=ablation --num_runs=1)
echo "Job assigned a row ID ${job_idx}"
echo "-----------"

ncsn_cfg=$(python ~/thesis_background/IsaacGymEnvs/isaacgymenvs/cfg/ablation_generator.py --model=ncsn --job_idx=${job_idx})
# ncsn_cfg=$(echo "$ncsn_output" | jq -r '.cmd')
# job_idx=$(echo "$ncsn_output" | jq -r '.job_idx')
# ncsn_cfg=$(python ~/thesis_background/IsaacGymEnvs/isaacgymenvs/utils/json_parser.py "${ncsn_output}" '.cmd')
# job_idx=$(python ~/thesis_background/IsaacGymEnvs/isaacgymenvs/utils/json_parser.py "${ncsn_output}" '.job_idx')

# Run ncsn if the cfg is not empty or done
if [ "${ncsn_cfg}" = "done" ]; then
  echo "cmds done!"
  echo "------------"
else
  echo "NCSN"
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
  echo "RL"
  echo ${rl_cfg}
  # python ~/thesis_background/IsaacGymEnvs/isaacgymenvs/train.py ${rl_cfg}
  
fi