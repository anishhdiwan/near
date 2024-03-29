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

import hydra

from omegaconf import DictConfig, OmegaConf

# Importing from the file path
import sys
import os
FILE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(FILE_PATH)


def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']

    train_cfg['device'] = cfg.rl_device

    train_cfg['population_based_training'] = cfg.pbt.enabled
    train_cfg['pbt_idx'] = cfg.pbt.policy_idx if cfg.pbt.enabled else None

    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    print(f'Using rl_device: {cfg.rl_device}')
    print(f'Using sim_device: {cfg.sim_device}')
    print(train_cfg)

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict


@hydra.main(version_base="1.1", config_name="gym_env_config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    import logging
    import os
    from datetime import datetime

    # noinspection PyUnresolvedReferences
    import isaacgym
    # from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
    # from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank
    from hydra.utils import to_absolute_path
    # from isaacgymenvs.tasks import isaacgym_task_map
    import gym
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed

    # if cfg.pbt.enabled:
    #     initial_pbt_check(cfg)

    # from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
    # from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs

    from isaacgymenvs.learning.diffusion_motion_priors import dmp_continuous


    # time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # run_name = f"{cfg.wandb_name}_{time_str}"

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
    from custom_envs.pusht_env import PushTEnv
    from custom_envs.particle_env import ParticleEnv
    from custom_envs.customenv_utils import CustomRayVecEnv, PushTAlgoObserver

    def create_env(**kwargs):
        if cfg_dict["task_name"] in ["pushT", "pushTAMP"]:
            env = PushTEnv(cfg=cfg_dict["task"]) # cfg is obtained from the config file. This is passed in within the algo init step as a kwarg
        elif cfg_dict["task_name"] in ["particle", "particleDMP"]:
            env = ParticleEnv(cfg=cfg_dict["task"])
        
        return env

    # env_configurations.register adds the env to the list of rl_games envs. create_isaacgym_env returns a VecTask environment. But rl_games also accepts gym envs. 
    env_configurations.register('gym_env', {
        'vecenv_type': 'CUSTOMRAY',
        'env_creator': lambda **kwargs: create_env(**kwargs),
    })

    # vecenv register calls the following lambda function which then returns an instance of CUSTOMRAY. 
    vecenv.register('CUSTOMRAY', lambda config_name, num_actors, **kwargs: CustomRayVecEnv(env_configurations.configurations, config_name, num_actors, **kwargs))


    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    # Build an rl_games runner. Register new AMP network builder and agent
    def build_runner():
        runner = Runner(algo_observer=PushTAlgoObserver())
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())


        ## Registering Diffusion Motion Priors ##
        runner.algo_factory.register_builder('dmp_continuous', lambda **kwargs : dmp_continuous.DMPAgent(**kwargs))

        return runner


    # create runner and set the settings
    runner = build_runner()
    runner.load(rlg_config_dict)
    runner.reset()


    # dump config dict
    if not cfg.test:
        experiment_dir = os.path.join('runs', cfg.train.params.config.name + 
        '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))

        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })


if __name__ == "__main__":
    launch_rlg_hydra()
