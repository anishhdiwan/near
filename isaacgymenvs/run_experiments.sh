#!/bin/bash

ncsn_cfg=$(python ~/thesis_background/IsaacGymEnvs/isaacgymenvs/cfg/experiment_generator.py --model=ncsn)

# Run ncsn if the cfg is not empty or done
if [ "${ncsn_cfg}" = "done" ]; then
  echo "cmds done!"
  echo "------------"
elif [ "${ncsn_cfg}" = "" ]; then
  echo "NCSN Skipped"
  echo "------------"
else
  # srun python ~/thesis_background/IsaacGymEnvs/isaacgymenvs/train_ncsn.py ${ncsn_cfg}
  echo ${ncsn_cfg}
fi

sleep 1.0

rl_cfg=$(python ~/thesis_background/IsaacGymEnvs/isaacgymenvs/cfg/experiment_generator.py --model=rl)

# Run rl (either AMP or NEAR) if not done
if [ "${rl_cfg}" = "done" ]; then
  echo "cmds done!"
  echo "------------"
else
  # srun python ~/thesis_background/IsaacGymEnvs/isaacgymenvs/train.py ${rl_cfg}
  echo ${rl_cfg}
fi