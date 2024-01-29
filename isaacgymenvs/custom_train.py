# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.










### A custom version of the isaacgymenvs train script adapted for non-isaacgym environment. Emulates the rl_games runner.py script ###


import hydra

from omegaconf import DictConfig, OmegaConf
from omegaconf import DictConfig, OmegaConf


# Hydra decorator to pass in the config. Looks for a config file in the specified path. This file in turn has links to other configs 
@hydra.main(version_base="1.1", config_name="custom_config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    import logging
    import os
    from datetime import datetime

    # noinspection PyUnresolvedReferences
    # import isaacgym
    from hydra.utils import to_absolute_path
    import gym
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed

    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    # from isaacgymenvs.learning import amp_continuous
    # from isaacgymenvs.learning import amp_players
    # from isaacgymenvs.learning import amp_models
    # from isaacgymenvs.learning import amp_network_builder
    # import isaacgymenvs

    # Naming the run
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.run_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)
    print("-----")

    # set numpy formatting for printing only
    set_np_formatting()
    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))
    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)


    # Creating a new function to return a pushT environment. This will then be added to rl_games env_configurations so that an env can be created from its name in the config
    from custom_envs.pusht_single_env import PushTEnv
    from custom_envs.customenv_utils import CustomRayVecEnv, PushTAlgoObserver

    def create_pusht_env(**kwargs):
        env =  PushTEnv()
        return env


    ### Explanation: env_configurations is a dictionary with env_name: dict(env config). This dict contains the type of vecenv and a creator function
    # vecenv.register is used to register a new environment TYPE and its creator function. For gym like environments, registering is most likely not
    # needed as they can simply use the existing TYPE "RAY". To do this, just add the env and its creator to env_configurations. This is what is done here.
    # Isaacgym environments also need a new env TYPE because they are not gym like. RLGPU is the registered name for these. 

    # env_configurations.register adds the env to the list of rl_games envs. create_isaacgym_env returns a VecTask environment. But rl_games also accepts gym envs. 
    env_configurations.register('pushT', {
        'vecenv_type': 'CUSTOMRAY',
        'env_creator': lambda **kwargs: create_pusht_env(**kwargs),
    })

    # vecenv register calls the following lambda function which then returns an instance of RLGPUEnv. 
    # In short, any new env must have the same functions as RLGPUEnv (or the same ones as in RayVecEnv from rl_games)
    # A simpler version of this is defined in this file as CustomEnv
    # once registered, the environment is instantiated automatically within the algorithm class in rl_games
    # vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: CustomEnv(config_name, num_actors, **kwargs))

    vecenv.register('CUSTOMRAY', lambda config_name, num_actors, **kwargs: CustomRayVecEnv(env_configurations.configurations, config_name, num_actors, **kwargs))

    # Convert to a big dictionary
    rlg_config_dict = omegaconf_to_dict(cfg.train)
 

    # Build an rl_games runner. You can add other algos and builders here
    def build_runner():
        runner = Runner(algo_observer=PushTAlgoObserver())
        return runner

    # create runner and set the settings
    runner = build_runner()
    runner.load(rlg_config_dict)
    runner.reset()

    # Run either training or playing via the rl_games runner
    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })


if __name__ == "__main__":
    launch_rlg_hydra()
