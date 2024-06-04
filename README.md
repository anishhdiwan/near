# Imitation Learning via Score/Energy-Based Generative Modelling 


## About this repository

This repository contains the implementation of a graduate thesis on imitation learning via score/energy-based generative modelling. This is a fork of the [Nvidia IsaacGym Environments](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) repository. It contains additional openAI_gym-based environments, the main code for this thesis, and baselines for comparison.

<br><br>


## Installation

**NOTE: This repository requires Ubuntu 20.04 and Python 3.7. These are hard requirements and are the latest possible Ubuntu/Python versions that are supported by Nvidia Isaac Gym**

This installation proceeds in three main steps. First, we install Isaac Gym, then we install the Isaacgymenvs (this repo) package. Finally we install the remaining packages required to train the proposed score/energy-based imitation learning algorithm.

1. Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then follow the installation instructions in the documentation. The conda environment installation option is required to run the code base from this project (other options might work but are not tested).

2. Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` directory, like `joint_monkey.py`. Follow the troubleshooting steps described in the Isaac Gym Preview 4 install instructions if you have any trouble running the samples. For instance, you might need to add the environment variable LD_LIB_PATH. 

3. Once Isaac Gym is installed and samples work within your current Python environment, install this repo:

```bash
conda install -c anaconda git
git clone --recurse-submodules -b <branch> https://github.com/anishhdiwan/diffusion_motion_priors.git
cd <dir-name>
pip install -e .
```

4. Finally, install the remaining dependencies:

```bash
pip install -r requirements.txt
```

5. If you plan to modify or visualise the Humanoid environment training data, you will also need to install the Autodesk .fbx Python SDK. Follow the instructions [here](https://help.autodesk.com/view/FBX/2020/ENU/?guid=FBX_Developer_Help_scripting_with_python_fbx_installing_python_fbx_html) to download the Python SDK. Once, installed, the Python package and libraries then need to be copied over to the conda environment to be accessible by other Python programs. Follow the instructions [here](https://download.autodesk.com/us/fbx/20112/fbx_sdk_help/index.html?url=WS73099cc142f48755-751de9951262947c01c-6dc7.htm,topicNumber=d0e8430) to do so. The whole procedure is also transcribed below. Note that you need to install the Python SDK and NOT the standard FBX SDK. The latest version that was tested with this repository is version 2020.2.1 for Python 3.7.

  - Download the Python SDK from [here](https://aps.autodesk.com/developer/overview/fbx-sdk)
  - Once downloaded, extract the archive and follow the instructions in the installation instructions file (it is recommended to make a new folder for the SDK to avoid clutter)
  - Once installed, navigate to the `lib` directory in the installation folder. Copy the contents of `<yourFBXSDKpath>\lib\<Pythonxxxxxx>\` to `conda\envs\rlgpu\lib\python3.7\site-packages\.`
  - NOTE: for the sdk to work, you need to add the LD_LIB_PATH environment variable `export LD_LIB_PATH=<conda path>/envs/rlgpu/lib`

<br><br>


## Training 

> [Hyperparams are explained here](hyperparameters.md)

**Training is a two-step procedure. First we train the energy-based model and subsequently use the trained model to learn a policy.**

**Note:** 
- Set params in the `dsm_config` part of the `train/<task-name>DMPPPO.yml` file. 
- The task data for CPU environments (mazeDMP, pushTDMP) is loaded automatically. The task data for the Humanoid environment is passed in the `motion_file` param of the `task/HumanoidDMP.yaml` file. This data is either in the `custom_envs/data/humanoid` directory or in the `assets` directory. Passing a .yaml file loads several motions while passing a single .npy file does single-clip training.
- Before training the policy, add the path to the trained energy-based model checkpoint in the `dsm_config` part of the `train/<task-name>DMP.yml` file.
- With IsaacGym, by default we show a preview window, which will usually slow down training. You can use the `v` key while running to disable viewer updates and allow training to proceed faster. Hit the `v` key again to resume viewing. Use the `esc` key or close the viewer window to stop training early. Alternatively, you can train headlessly by adding the headless:True argument. 

### Step 1: Training the energy-based model
```bash
# tasks = [mazeDMP, pushTDMP, HumanoidDMP]
python train_ncsn.py task=<task-name>
```

### Step 2: Training the energy-based policy

```bash
# For CPU-based environments
# tasks = [mazeDMP, pushTDMP]
python train_gym_envs.py task=<task-name> headless=<bool>  
```

```bash
# For Isaac Gym environments
# tasks = [HumanoidDMP]
python train.py task=<task-name> headless=<bool>  
```

<br><br>


## Training Baselines 

**Baselines such as Adversarial Motion Priors and Cross Entropy Method (currently only for CPU envs) are trained similarly to the previous procedure. However, they do not need the first step of training the energy based model. Training can be done as follows.**


```bash
# For CPU-based environments
# tasks = [mazeAMP, mazeCEM, pushTAMP, pushTCEM, pushT, maze]
# If no algo is mentioned, then PPO is used by default (example: pushT trains with PPO while pusTAMP trains with AMP)
# For CEM, also mention the train script separately (example: python train_gym_envs.py task=mazeCEM train=mazeCEM)
python train_gym_envs.py task=<task-name> headless=<bool>  
```

```bash
# For Isaac Gym environments
# tasks = [HumanoidAMP, Humanoid]
# If no algo is mentioned, then PPO is used by default (example: Humanoid trains with PPO while HumanoidAMP trains with AMP)
python train.py task=<task-name> headless=<bool>  
```

<br><br>


## Visualising Trained Policies

Trained policies can be visualised as follows

```bash
# For CPU-based environments
# tasks = [mazeAMP, pushTAMP, pushT, maze, mazeDMP, pushTDMP]
# If no algo is mentioned, then PPO is assumed by default
# Make sure to set the visualise_disc argument of in tasks/mazeAMPPPO.yaml to False
python train_gym_envs.py task=<task-name> test=True checkpoint=<path-to-saved-checkpoint>
```

```bash
# For Isaac Gym environments
# tasks = [HumanoidAMP, Humanoid, HumanoidDMP]
# If no algo is mentioned, then PPO is assumed by default
python train.py task=<task-name> test=True checkpoint=<path-to-saved-checkpoint> 
```


### Visualising the energy function or AMP discriminator
Since the maze environment is 2-dimensional, the energy-function or the adversarial motion priors discriminator can be visualised. 

```bash
# Visualising the energy function
python train_ncsn.py task=mazeDMP test=True
```

```bash
# Visualising the discriminator
# Make sure to set the visualise_disc argument of in tasks/mazeAMPPPO.yaml to True
python train_gym_envs.py task=mazeAMP test=True checkpoint=<path-to-saved-checkpoint>
```

<br><br>

## Datasets
**NOTE: The data used in this project was obtained from [http://mocap.cs.cmu.edu/](http://mocap.cs.cmu.edu/). The database was created with funding from NSF EIA-0196217**

Processed demonstration data for training both AMP and the proposed approach is also available in the repository. This data was obtained from the CMU mo-cap dataset, then filtered and processed. The Adversarial Motion Priors codebase provides useful tools to process this data. This repository extends these tools. To process your own demonstration data first obtain the dataset in the .fbx format.

1. The `isaacgymenvs/tasks/amp/poselib` directory houses some scripts for data processing. Use the `fbx_motion_to_npy.py` script to convert the .fbx dataset into .npy files
2. Then use the `generate_retargeted_dataset.py` script to retarget these motions from the CMU skeleton to the .mjcf skeleton used for experiments in the work.
3. You might have to first obtain a skeleton to retarget the data. Use `generate_tpose_from_motion.py` to generate a skeleton .npy file based on which motions are retargeted.

The final motions are then placed in the `isaacgymenvs/custom_envs/data/humanoid` directory and a .yaml file is created to pass them all together to the learning algorithm.

<br><br>

## Troubleshooting

Please review the Isaac Gym installation instructions first if you run into any issues.

You can either submit issues through GitHub or through the [Isaac Gym forum here](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/isaac-gym/322).

## Citing
