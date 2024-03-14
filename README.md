# Diffusion Motion Priors - Imitation Learning via Score-Based Generative Modelling 


### About this repository

This repository contains the implementation of the thesis on imitation learning via score-based generative modelling. This is a fork of the [Nvidia IsaacGym Environments](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) repository. It containes additional openAI gym based environments, the main code for this thesis, and additional baselines for comparison.


### Installation

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions in the documentation. We highly recommend using a conda environment 
to simplify set up.

Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` 
directory, like `joint_monkey.py`. Follow troubleshooting steps described in the Isaac Gym Preview 4
install instructions if you have any trouble running the samples.

Once Isaac Gym is installed and samples work within your current python environment, install this repo:

```bash
pip install -e .
```


### Running the benchmarks

#### OpenAIGym Envs 

Currently only the pushT environment is included.

PPO
```bash
python train_gym_envs.py task=pushT
```

Adversarial Motion Priors
```bash
python train_gym_envs.py task=pushTAMP
```

#### IsaacGym Envs

PPO
```bash
python train.py task=Cartpole
```

PPO
```bash
python train.py task=HumanoidAMP experiment=AMP_walk
```


Note that with IsaacGym, by default we show a preview window, which will usually slow down training. You 
can use the `v` key while running to disable viewer updates and allow training to proceed 
faster. Hit the `v` key again to resume viewing after a few seconds of training, once the 
ants have learned to run a bit better.

Use the `esc` key or close the viewer window to stop training early.

Alternatively, you can train headlessly, as follows:

```bash
python train.py task=Cartpole headless=True
```


## Troubleshooting

Please review the Isaac Gym installation instructions first if you run into any issues.

You can either submit issues through GitHub or through the [Isaac Gym forum here](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/isaac-gym/322).

## Citing