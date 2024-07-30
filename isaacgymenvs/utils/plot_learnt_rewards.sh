#!/bin/bash

ALGO="AMP"
CHECKPOINTS="3932160 9830400 19660800 29491200 39321600 58982400"
TRIALS="HumanoidAMP_walk_8125 HumanoidAMP_walk_700 HumanoidAMP_walk_42 HumanoidAMP_walk_97 HumanoidAMP_walk_3538"

# ALGO="NEAR"
# CHECKPOINTS="-1"
# TRIALS="HumanoidNEAR_cartwheel_700_ncsnv2_db"

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


