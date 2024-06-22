#!/bin/bash

# ALGO="AMP"
# CHECKPOINTS="0 5000"
# TRIALS="HumanoidAMP_walk_42 HumanoidAMP_walk_43"

ALGO="DMP"
CHECKPOINTS="-1"
TRIALS="Humanoid_SM_temporal_states_walk"

play_cmd=$(python ./utils/plot_learnt_rewards.py --algo ${ALGO} --trials ${TRIALS} --checkpoints ${CHECKPOINTS})

while [ "${play_cmd}" != "done" ]
do 
#   echo ${play_cmd}
  python ${play_cmd}
  sleep 1.0

  play_cmd=$(python ./utils/plot_learnt_rewards.py --algo ${ALGO} --trials ${TRIALS} --checkpoints ${CHECKPOINTS})
  

done


