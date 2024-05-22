# Imitation Learning via Score/Energy-Based Generative Modelling 


### About this repository

This repository contains the implementation of a graduate thesis on imitation learning via score/energy-based generative modelling. This is a fork of the [Nvidia IsaacGym Environments](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) repository. It contains additional openAI_gym-based environments, the main code for this thesis, and baselines for comparison.


### Installation

Ubuntu 20.04 is necessary to run this project (Isaac Gym requires Ubuntu 20.04). This installation proceeds in three main steps. First, we install Isaac Gym, then we install the Isaacgymenvs (this repo) package. Finally we install the remaining packages required to train the proposed score/energy-based imitation learning algorithm.

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

  -- Download the Python SDK from [here](https://aps.autodesk.com/developer/overview/fbx-sdk)
  -- Once downloaded, extract the archive and follow the instructions in the readme file (it is recommended to make a new folder for the SDK to avoid clutter)
  -- Once installed, navigate to the `lib` directory in the installation folder. Copy the contents of <yourFBXSDKpath>\lib\<Pythonxxxxxx>\ to conda\envs\rlgpu\lib\python3.7\site-packages\.
  -- NOTE: for the sdk to work, you need to add the environment variable LD_LIB_PATH:=<conda path>/envs/rlgpu/lib


### Running the benchmarks

Note: [hyperparams are explained here](hyperparameters.md)

#### OpenAIGym Envs 

PPO
```bash
python train_gym_envs.py task=pushT
```

Adversarial Motion Priors
```bash
python train_gym_envs.py task=pushTAMP
```

Cross Entropy Method
```bash
python train_gym_envs.py task=mazeCEM train=mazeCEM
```

possible tasks = ['pushT', 'mazeEnv']

#### IsaacGym Envs

PPO
```bash
python train.py task=Cartpole
```

PPO
```bash
python train.py task=HumanoidAMP experiment=AMP_walk
```

Note: with IsaacGym, by default we show a preview window, which will usually slow down training. You 
can use the `v` key while running to disable viewer updates and allow training to proceed 
faster. Hit the `v` key again to resume viewing after a few seconds of training, once the 
ants have learned to run a bit better.

Use the `esc` key or close the viewer window to stop training early.

Alternatively, you can train headlessly, as follows:

```bash
python train.py task=Cartpole headless=True
```

### Training the energy-based model (NCSN)

```bash
python train_ncsn.py task=particleDMP
```

Note: set DMP params in the `dsm_config` part of the `train/particleDMP.yml` file. The motions are currently preset within the ncsn script but will soon be passable as config params. 

### Visualising the energy function or AMP discriminator

```bash
python train_ncsn.py task=particleDMP test=True
```

```bash
python train_gym_envs.py task=mazeDMP test=True checkpoint=runs/run_name/nn/mazeAMP_datetime.pth
```





## Troubleshooting

Please review the Isaac Gym installation instructions first if you run into any issues.

You can either submit issues through GitHub or through the [Isaac Gym forum here](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/isaac-gym/322).

## Citing
