#!/bin/bash

# NOTE: Place this in /scratch
for i in 1 2
do
   /bin/bash ~/near/isaacgymenvs/send_db_job.sh
   echo "Job $i Sent!"
   echo "-----"
done