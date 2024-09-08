#!/bin/bash
#SBATCH --job-name="NEAR_experiments"
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=gpu-a100
#SBATCH --account=research-me-cor

# Load modules:
module load 2023r1
module load openmpi
module load miniconda3

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
conda activate rlgpu
export LD_LIBRARY_PATH=/home/adiwan/.conda/envs/rlgpu/lib/python3.7/site-packages/nvidia/cublas/lib/:/home/adiwan/.conda/envs/rlgpu/lib
cd /scratch/adiwan

#####
job_idx=$1
echo "Job assigned a row ID ${job_idx}"
echo "-----------"


ncsn_cfg0=$(python ~/near/isaacgymenvs/cfg/experiment_generator.py --model=ncsn --job_idx=${job_idx} --kth_ncsn=0)
ncsn_cfg1=$(python ~/near/isaacgymenvs/cfg/experiment_generator.py --model=ncsn --job_idx=${job_idx} --kth_ncsn=1)
# ncsn_cfg=$(echo "$ncsn_output" | jq -r '.cmd')
# job_idx=$(echo "$ncsn_output" | jq -r '.job_idx')
# ncsn_cfg=$(python ~/near/isaacgymenvs/utils/json_parser.py "${ncsn_output}" '.cmd')
# job_idx=$(python ~/near/isaacgymenvs/utils/json_parser.py "${ncsn_output}" '.job_idx')

# Run ncsn if the cfg is not empty or done
if [ "${ncsn_cfg0}" = "done" ]; then
  echo "cmds done!"
  echo "------------"
elif [ "${ncsn_cfg0}" = "" ]; then
  echo "NCSN Skipped"
  echo "------------"
else
  echo "NCSN"
  echo ${ncsn_cfg0}
  echo ${ncsn_cfg1}
  srun python ~/near/isaacgymenvs/train_ncsn.py ${ncsn_cfg0} &
  srun python ~/near/isaacgymenvs/train_ncsn.py ${ncsn_cfg1} &
fi

wait
sleep 1.0

rl_cfg=$(python ~/near/isaacgymenvs/cfg/experiment_generator.py --model=rl --job_idx=${job_idx})

# Run rl (either AMP or NEAR) if not done
if [ "${rl_cfg}" = "done" ]; then
  echo "cmds done!"
  echo "------------"
else
  echo "RL"
  echo ${rl_cfg}
  srun python ~/near/isaacgymenvs/train.py ${rl_cfg}
fi

#####

wait
conda deactivate
