#!/bin/bash

# ALGO="AMP"
# CHECKPOINTS="0 5000"
# TRIALS="HumanoidAMP_walk_42 HumanoidAMP_walk_43"

ALGO="DMP"
CHECKPOINTS="-1"
TRIALS="plottest1 plottest2"

play_cmd=$(python ./utils/plot_learnt_rewards.py --algo ${ALGO} --trials ${TRIALS} --checkpoints ${CHECKPOINTS})

while [ "${play_cmd}" != "done" ]
do 
  echo "---------"
  echo ${play_cmd}
  echo "---------"
  sleep 0.5
  python ${play_cmd}
  sleep 1.0

  play_cmd=$(python ./utils/plot_learnt_rewards.py --algo ${ALGO} --trials ${TRIALS} --checkpoints ${CHECKPOINTS})
  

done


