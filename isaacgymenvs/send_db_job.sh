#!/bin/bash
#SBATCH --job-name="NEAR_experiments"
#SBATCH --time=00:45:00
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

ncsn_cfg=$(python ~/near/isaacgymenvs/cfg/experiment_generator.py --model=ncsn)

# Run ncsn if the cfg is not empty or done
if [ "${ncsn_cfg}" = "done" ]; then
  echo "cmds done!"
  echo "------------"
elif [ "${ncsn_cfg}" = "" ]; then
  echo "NCSN Skipped"
  echo "------------"
else
  echo ${ncsn_cfg}
  srun python ~/near/isaacgymenvs/train_ncsn.py ${ncsn_cfg}
fi

sleep 1.0

rl_cfg=$(python ~/near/isaacgymenvs/cfg/experiment_generator.py --model=rl)

# Run rl (either AMP or NEAR) if not done
if [ "${rl_cfg}" = "done" ]; then
  echo "cmds done!"
  echo "------------"
else
  echo ${rl_cfg}
  srun python ~/near/isaacgymenvs/train.py ${rl_cfg}
fi

#####

wait
conda deactivate

