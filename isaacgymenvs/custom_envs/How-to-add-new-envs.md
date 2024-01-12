# Introduction to [rl_games](https://github.com/Denys88/rl_games/)  - new envs, and new algorithms built on rl_games
This write-up describes some elements of the general functioning of the [rl_games](https://github.com/Denys88/rl_games/) reinforcement learning library and provides a guide on adding new environments to rl_games, using non-gym environments and using simulators with the algorithms in rl_games, and modifying the existing algorithms in rl_games.

## General setup in rl_games

rl_games uses the main python script called `runner.py` along with flags for either training (`--train`) or executing policies (`--play`) and a mandatory option for training/playing configurations `--file`. A basic example of training and then playing for PPO in Pong can be executed with 

```
python runner.py --train --file rl_games/configs/atari/ppo_pong.yaml
python runner.py --play --file rl_games/configs/atari/ppo_pong.yaml --checkpoint nn/PongNoFrameskip.pth
```

As you might have noticed, logs and trained checkpoints are saved in a directory called nn. The rest of this tutorial builds on this setup from rl_games but uses [hydra](https://hydra.cc/docs/intro/) for easier configuration management. Further, instead of directly using `runner.py` we use another similar script called `train.py` which allows us to dynamically add new environments and insert out own algorithms. This modified setup comes from [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs), the NVIDIA repository for RL simulations and training. 

With all this considered, our final file structure is something like this.

```
project dir
│   train.py    
│
└───tasks dir (sometimes also called envs dir)
│   │   customenv.py
│   │   rl_games_env_utils.py
│   │
│   └───subfolder1
│       │   file111.txt
│       │   file112.txt
│       │   ...
|
└───cfg dir (main hydra configs)
│   │   config.yaml (main config for the setting up simulators etc. if needed)
│   │
│   └─── task dir (configs for the env)
│       │   customenv.yaml
│       │   otherenv.yaml
│       │   ...
|   
│   └─── train dir (configs for training the algorithm)
│       │   customenvAlgo.yaml
│       │   otherenvPPO.yaml
│       │   ...
|
└───algos dir (custom wrappers for training algorithms in rl_games)
|   │   custom_network_builder.py
|   │   custom_algo.py
|   | ...
|
└───runs dir (generated automatically on running train.py)
│   └─── env_name_alg_name_datetime dir (train logs)
│       └─── nn
|           |   checkpoints.pth
│       └─── summaries
            |   events.out...
```

rl_games uses the following base classes to define algorithms, instantiate environments, and log metrics.

1. `rl_games.torch_runner.Runner` 
    - Main class that instantiates the algorithm as per the given configuration and executes either training or playing 
    - When instantiated, algorithm instances for all algos in rl_games are automatically added using `rl_games.common.Objectfactory()`'s `register_builder()` method. The same is also done for the player instances for all algos. 
    - Adding a custom algorithm essentially translates to registering your own builder and player
    - Depending on the args given, either `self.run_train()` or `self.run_play()` is executed 
    - Also sets up the algorithm observer that logs training metrics. If one is not provided, it automatically uses the `DefaultAlgoObserver()` which simply logs metrics using the tensorboard summarywriter. Custom observers can also be provided based on your requirements.

2. `rl_games.common.Objectfactory()`
    - Creates algorithms or players. Has the `register_builder(self, name, builder)` method that given a string name adds a function that returns whatever is being built. For example the following line adds the name a2c_continuous to a lambda function that returns the A2CAgent
        ```
        register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        ```
    - Also has a `create(self, name, **kwargs)` method that simply returns one of the registered builders by name \

3. `RL Algorithms`
    - rl_games has several reinforcement learning algorithms. Most of these inherit from some sort of base algorithm class, for example, `rl_games.algos_torch.A2CBase`. In rl_games environments are instantiated by the algorithm. 

4. `Environments (rl_games.common.vecenv & rl_games.common.env_configurations)`
    - The vecenv script holds classes to instantiate different environments based on their type. Since rl_games is quite a broad library, it supports multiple environment types (such as openAI gym envs, brax envs, cule envs etc). These environment types and their base classes are stored in the `rl_games.common.vecenv.vecenv_config` dictionary. A new environment type can be added by calling the `vecenv.register(config_name, func)` function that simply adds the `config_name:func` pair to the dictionary. For example the following line adds a 'RAY' type env with a lambda function that then instantiated the RayVecEnv class. NOTE: most gym-like environments can be instantiated with the RAY env type. The "RayVecEnv" holds "RayWorkers" that internally store the environment. This automatically allows for multi-env training.
        ```
        register('RAY', lambda config_name, num_actors, **kwargs: RayVecEnv(config_name, num_actors, **kwargs))
        ```

    - `rl_games.common.env_configurations` is another dictionary that stores `env_name: {'vecenv_type', 'env_creator}` information. For example, the following stores the environment name "CartPole-v1" with a value for its type and a lambda function that instantiates the respective gym env.
        ```    
        'CartPole-v1' : {
            'vecenv_type' : 'RAY',
            'env_creator' : lambda **kwargs : gym.make('CartPole-v1'),}
        ```
    - The general idea here is that the algorithm base class instantiates a new environment by looking at the env_name (for example 'CartPole-v1') in the config file. Internally, the name 'CartPole-v1' is used to get the env type from `rl_games.common.env_configurations`. The type then goes into the `vecenv.vecenv_config` dict which returns the actual environment class (such as RayVecEnv)
    - Note, the env class (such as RayVecEnv) then internally uses the 'env_creator' key to instantiate the environment using whatever function was given to it (for example, `lambda **kwargs : gym.make('CartPole-v1')`)


## Adding new gym-like environments
Adding a gym-like environment essentially translates to creating a gym-like env class (that inherits from gym.Env) and adding this under the type 'RAY' to `rl_games.common.env_configurations`. Ideally, this needs to be done by adding the key value pair `env_name: {'vecenv_type', 'env_creator}` to `env_configurations.configurations`. However, this requires modifying the rl_games library. If you do not wish to do that then you can instead use the register method to add your new env to the dictionary, then make a copy of the RayVecEnv and RayWorked classes and change the __init__ method to instead take in the modified env configurations dict. 


For example

```
def create_new_env(**kwargs):
    # Instantiate new env
    env =  newEnv()

    #For example, env = gym.make('LunarLanderContinuous-v2')
    return env

from rl_games.common import env_configurations, vecenv

env_configurations.register('customenv', {
    'vecenv_type': 'COPYRAY',
    'env_creator': lambda **kwargs: create_custom_env(**kwargs),
})

vecenv.register('COPYRAY', lambda config_name, num_actors, **kwargs: CustomRayVecEnv(env_configurations.configurations, config_name, num_actors, **kwargs))

------------

# Make a copy of RayVecEnv

class CustomRayVecEnv(IVecEnv):
    import ray

    def __init__(self, config_dict, config_name, num_actors, **kwargs):
        # Explicityly passing in the dictionary containing env_name: {vecenv_type, env_creator}
        self.config_dict = config_dict

        self.config_name = config_name
        self.num_actors = num_actors
        self.use_torch = False
        self.seed = kwargs.pop('seed', None)

        
        self.remote_worker = self.ray.remote(CustomRayWorker)
        self.workers = [self.remote_worker.remote(self.config_dict, self.config_name, kwargs) for i in range(self.num_actors)]

        ...
        ...

# Make a copy of RayWorker

class CustomRayWorker:
    def __init__(self, config_dict, config_name, config):
        self.env = config_dict[config_name]['env_creator'](**config)

        ...
        ...
```

## Adding non-gym environments & simulators

## New algorithms within rl_games

