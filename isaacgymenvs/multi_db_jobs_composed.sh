#!/bin/bash

# NOTE: Place this in /scratch
job_ids=$(python ~/near/isaacgymenvs/utils/plan_cmds.py --run_type=experiment --num_runs=5)

if [ "${job_ids}" = "None" ]; then
  echo "cmds done!"
  echo "------------"
else  
   for i in ${job_ids}
   do
      sbatch ~/near/isaacgymenvs/send_db_job_composed.sh ${i}
      echo "Job $i Sent!"
      echo "-----"
   done
fi
