# Training hyperparameters

Note: This repository is an extension of [rl_games](https://github.com/Denys88/rl_games/tree/master) and [isaacgymenvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) for work on imitation learning agents. Most of the parameters here are also found in the parent repositories. However, this doc provides more detailed definitions of these training hyperparameters and suggestions for their values depending on the task at hand. View the set of params used for training [here](#set-of-params) (please refer to the parent repositories for an exhaustive list of params). The full yaml config can be found in the cfg directory.


## Definitions & suggested values
| **PARAM** 	| **DEFINITION** 	| **SRC CODE URL** 	|
|---	|---	|---	|
| **_algo_** 	|  	|  	|
| name 	| Algorithm name. This is used to build the algorithm class<br><br>Possible values are: sac, a2c_discrete, a2c_continuous<br><br>Additional algos: amp_continuous, dmp_continuous, cem<br> 	|  	|
|  	|  	|  	|
| **_model_** 	|  	|  	|
| name 	| Model name. The model internally instantiates the network. The network runs the actual neural net computations while the model processes these to return actions, values, etc in a dict. format. In most cases, the model returns actions by sampling from a normal distribution with learnt mean and covariance.<br><br>Possible values: continuous_a2c ( expects sigma to be (0, +inf), continuous_a2c_logstd ( expects sigma to be (-inf, +inf), a2c_discrete, a2c_multi_discrete 	| https://github.com/Denys88/rl_games/blob/cba782ceb772795628e52a3da3d5dc8c20ecb779/rl_games/algos_torch/model_builder.py#L35<br><br>https://github.com/Denys88/rl_games/blob/master/rl_games/algos_torch/models.py 	|
|  	|  	|  	|
| **_network_** 	|  	|  	|
| name 	| Network name. The network is the actual neural network that takes in observations to return a set of quantities (values, means, covariances ...)<br><br>Possible values: actor_critic or soft_actor_critic. 	| https://github.com/Denys88/rl_games/blob/master/rl_games/algos_torch/model_builder.py 	|
| separate 	| Whether to use separate networks for the actor and critic 	|  	|
| _space:continuous (discrete spaces are not covered here but you can refer to rl_games)_ 	|  	|  	|
| mu_activation 	| Activation function for the mean computation within the network 	| Possible values: https://github.com/Denys88/rl_games/blob/cba782ceb772795628e52a3da3d5dc8c20ecb779/rl_games/algos_torch/network_builder.py#L34 	|
| sigma_activation 	| Activation function for the covariance computation within the network 	| Possible values: https://github.com/Denys88/rl_games/blob/cba782ceb772795628e52a3da3d5dc8c20ecb779/rl_games/algos_torch/network_builder.py#L34 	|
| fixed_sigma 	| Whether to learn the covariance or have it be fixed. If False, then sigma is learnt using a neural network 	|  	|
| _mu_init_ 	|  	|  	|
| name 	| Name of the initialisation procedure for the mean. This is applied as a transformation on the mean in the network 	| Possible values: https://github.com/Denys88/rl_games/blob/cba782ceb772795628e52a3da3d5dc8c20ecb779/rl_games/algos_torch/network_builder.py#L45 	|
| scale 	| kwarg value pertaining to the initialisation. For example, with const_initializer, the tensor is initialised to have all elements to this value 	|  	|
| _sigma_init_ 	|  	|  	|
| name 	| Name of the initialisation procedure for the covariance. This is applied as a transformation on the mean in the network 	| Possible values: https://github.com/Denys88/rl_games/blob/cba782ceb772795628e52a3da3d5dc8c20ecb779/rl_games/algos_torch/network_builder.py#L45 	|
| scale 	| kwarg value pertaining to the initialisation. For example, with const_initializer, the tensor is initialised to have all elements to this value<br><br>NOTE: Most models use exp(sigma) as the covariance matrix of the Gaussian from which actions are sampled. When using const_initializer with fixed_sigma=True, the scale is the value to which the covariance is set and no learning is done. In this case, if scale=0.0 then actions are sampled from a distributions with the learnt mean and identity covariance. This is alright if the environment accepts unnormalised actions. HOWEVER, if the actions are normalised (in range -1,1) in the environment then under this config, learning would be very noisy as the model would only sample actions from the whole action space. Setting it to -ve values is preferred if your actions are normalised in the environment 	|  	|
| _mlp_ 	|  	|  	|
| units 	| Array of sizes of the MLP layers, for example: [512, 256, 128] 	|  	|
| activation 	| MLP activation layer 	| Possible values: https://github.com/Denys88/rl_games/blob/cba782ceb772795628e52a3da3d5dc8c20ecb779/rl_games/algos_torch/network_builder.py#L34 	|
| _initializer_ 	|  	|  	|
| name 	| Name of the initialization procedure for the MLP 	| Possible values: https://github.com/Denys88/rl_games/blob/cba782ceb772795628e52a3da3d5dc8c20ecb779/rl_games/algos_torch/network_builder.py#L45 	|
| scale 	| kwarg value pertaining to the initialisation. For example, with const_initializer, the tensor is initialised to have all elements to this value 	|  	|
|  	|  	|  	|
| **_load_checkpoint_** 	| Whether to begin training from a certain checkpoint 	|  	|
| **_load_path_** 	| Path to the saved checkpoint 	|  	|
|  	|  	|  	|
| **_config (config for the training algo)_** 	|  	|  	|
| name: 	| Experiment name 	|  	|
| full_experiment_name: 	| Experiment name override 	|  	|
| max_epochs: 	| Max training epochs 	|  	|
| env_name: 	| Name of the environment type used in rl_games. Refer to https://github.com/Denys88/rl_games/blob/master/docs/HOW_TO_RL_GAMES.md  	|  	|
| normalize_advantage: 	| Whether to normalise computed advantages 	|  	|
| gamma: 	| Discount factor 	|  	|
| tau: 	| Lambda for generalized advantage estimation. Called tau by mistake long time ago because lambda is keyword in python<br><br>GAE Ref<br>John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel. 2015.<br>High-Dimensional Continuous Control Using Generalized Advantage Estimation.<br>CoRR abs/1506.02438 (2015). arXiv:1506.02438 	|  	|
| learning_rate: 	| Learning rate 	|  	|
| score_to_win: 	| Stop training when moving avg of env return reaches this 	|  	|
| save_best_after: 	| Start saving the best model after these many epochs 	|  	|
| save_frequency: 	| Save at this rate 	|  	|
| grad_norm: 	| Max norm (L2) for gradient truncation. Applied if truncate_grads is True. 	|  	|
| truncate_grads: 	| Whether to truncate gradients of the network based on their norm 	|  	|
| entropy_coef: 	| Starting entropy coefficient for the learning rate scheduler (if lr_schedule=linear) 	|  	|
| e_clip: 	| Clip parameter for ppo loss. 	|  	|
| clip_value: 	| Whether to apply clip to the value loss. If you are using normalize_value you don't need it. 	|  	|
| num_actors: 	| Number of parallel environments (NOT the number of agents in an env. Refer to num_agents for this) 	|  	|
| horizon_length: 	| Length of the horizon. The agent applies the current policy for these many steps, then evaluates the advantages, and updates the policy 	|  	|
| minibatch_size: 	| Used to define the size of the minibatch used to update the network. Used as a parameter within the dataset class.<br><br>Most algorithms use a batch size of horizon_len * num_actors * num_agents. The minibatch size is a parameter that must be exactly divisible by this value. This ensures that a batch can exactly comprise of whole number mini batches  	| https://github.com/Denys88/rl_games/blob/cba782ceb772795628e52a3da3d5dc8c20ecb779/rl_games/common/datasets.py#L8 	|
| mini_epochs: 	| Number of times to update the network in each epoch 	|  	|
| critic_coef: 	| Critic coefficient 	|  	|
| lr_schedule: 	| Learning rate scheduling policy, Could be None, linear or adaptive. 	|  	|
| kl_threshold: 	| KL threshould for adaptive schedule. if KL < kl_threshold/2 lr = lr * 1.5 and opposite. 	|  	|
| normalize_input: 	| Normalize the network input (observations) 	|  	|
| normalize_value: 	| Normalize the state/action values predicted by the network 	|  	|
| value_bootstrap: 	| Whether to bootstrap values when the episode finishes 	|  	|
| bounds_loss_coef: 	| Regularisation coefficient 	|  	|
| mixed_precision: 	| Whether to use the torch mixed precision package to scale gradients. Default=False 	| https://pytorch.org/docs/stable/amp.html 	|
| print_stats: 	| Whether to print training stats 	|  	|
| reward_shaper 	| Simple transformations on the environments reward function 	|  	|
| scale_value 	| Value to scale by 	|  	|
|  	|  	|  	|
| ######################### 	| #################################################### 	| #################### 	|
| ######################### 	| #################################################### 	| #################### 	|
|  	|  	|  	|
| **AMP** 	|  	|  	|
| amp_obs_demo_buffer_size: 	| Demo observations buffer size 	|  	|
| amp_replay_buffer_size:  	| Replay buffer size 	|  	|
| amp_replay_keep_prob:  	| Probability of keeping observations in the buffer 	|  	|
| amp_batch_size:  	| Batch size 	|  	|
| amp_minibatch_size:  	| Minibatch size 	|  	|
| disc_coef:  	| Discriminator loss coefficient 	|  	|
| disc_logit_reg:  	| Discriminator output regularisation 	|  	|
| disc_grad_penalty:  	| Disciminator gradient penalty coefficient 	|  	|
| disc_reward_scale:  	| Discriminator reward scaling 	|  	|
| disc_weight_decay:  	| Discriminator weight decay 	|  	|
| normalize_amp_input:  	| Whether to normalise observations fed to the discriminator 	|  	|
|  	|  	|  	|
| ######################### 	| #################################################### 	| #################### 	|
| ######################### 	| #################################################### 	| #################### 	|
|  	|  	|  	|
| **NEAR** 	|  	|  	|
|  	|  	|  	|
| **_dmp_config:_** 	|  	|  	|
| _training:_ 	|  	|  	|
| batch_size:  	| NCSN batch size 	|  	|
| buffer_size:  	| NCSN buffer size when using humanoid data. This samples a buffer of data from the humanoid demonstrations and then subsamples batches from that data 	|  	|
| n_epochs:  	| Number of training epochs (set to an upper bound and use n_iters to early stop) 	|  	|
| n_iters:  	| Number of iters to stop after 	|  	|
| ngpu:  	| Number of GPUs. Set to 1 	|  	|
| snapshot_freq:  	| How often to save checkpoints 	|  	|
| algo:  	| Which score-matching algo to use. Set to 'dsm' 	|  	|
| anneal_power:  	| Additional regularisation used with the DSM loss. Adds sigma ** power to the loss 	|  	|
| normalize_energynet_input:  	| Whether to normalise observations fed to NCSN 	|  	|
|  	|  	|  	|
| _data:_ 	|  	|  	|
| dataset:  	| Dataset 	|  	|
| motion_file:  	| Motion file (fed from the task .yaml file) 	|  	|
|  	|  	|  	|
| _model:_ 	|  	|  	|
| sigma_begin:  	| Max noise sigma 	|  	|
| sigma_end:  	| Min noise sigma 	|  	|
| L:  	| Number of noise levels 	|  	|
| in_dim:  	| Number of features in the input observation 	|  	|
| encoder_hidden_layers:  	| Encoder dims 	|  	|
| latent_space_dim:  	| Latent space dims 	|  	|
| decoder_hidden_layers:  	| Decoder dims (the out dim is set to 1 internally) 	|  	|
| ema: 	| Whether to track the expontentially moving average of model weights  	|  	|
| ema_rate: 	| The EMA rate 	|  	|
| ncsnv2: 	| Whether to use the improved NCSN algorithm 	|  	|
|  	|  	|  	|
| _optim:_ 	|  	|  	|
| weight_decay:  	| Optimiser weight decay 	|  	|
| optimizer:  	| Optimiser type. Set to 'Adam' 	|  	|
| lr:  	| Learning rate 	|  	|
| beta1: 	| Beta value 	|  	|
| amsgrad:  	| Whether to use amsgrad 	|  	|
|  	|  	|  	|
| _inference:_ 	|  	|  	|
| task_reward_w:  	| Task reward weightage 	|  	|
| energy_reward_w:  	| Energy reward weightage 	|  	|
| sigma_level:  	| Noise level to use to train the policy. Must be in range [0,L] or -1. <br><br>If -1 then annealing is used 	|  	|
| eb_model_checkpoint:  	| Saved NCSN checkpoint relative to train.py 	|  	|
| running_mean_std_checkpoint:  	| Saved running mean checkpoint relative to train.py 	|  	|

## Set of params



### PPO Params
```yaml
params: 
 
  algo:
    name: 

  model:
    name: 

  network:
    name: 
    separate: 
    space:
      continuous:
        mu_activation: 
        sigma_activation: 
        mu_init:
          name: 
          scale: 
        sigma_init:
          name: 
          val: 
        fixed_sigma: 
    mlp:
      units: 
      activation: 
      initializer:
          name: 
          scale: 

  load_checkpoint: 
  load_path: 

  config:
    name: 
    full_experiment_name:
    max_epochs: 
    env_name:  
    
    reward_shaper:
      scale_value: 
    
    normalize_advantage: 
    gamma: 
    tau: 
    learning_rate: 
    score_to_win: 
    save_best_after: 
    save_frequency: 
    grad_norm: 
    entropy_coef: 
    truncate_grads: 
    e_clip: 
    clip_value: 
    num_actors: 
    horizon_length: 
    minibatch_size: 
    mini_epochs: 
    critic_coef: 
    lr_schedule:  
    kl_threshold: 
    normalize_input: 
    normalize_value: 
    value_bootstrap: 
    bounds_loss_coef: 
    mixed_precision: 
    print_stats: 

    player:
      render: 
      render_sleep: 
      games_num: 
```


### AMP - Additional Params

```yaml
amp_obs_demo_buffer_size: 
amp_replay_buffer_size: 
amp_replay_keep_prob: 
amp_batch_size: 
amp_minibatch_size: 
disc_coef: 
disc_logit_reg: 
disc_grad_penalty: 
disc_reward_scale: 
disc_weight_decay: 
normalize_amp_input: 
task_reward_w: 
disc_reward_w: 
```

### NEAR - Additional Params

```yaml
near_config:
  
  training:
    batch_size: 
    buffer_size: 
    n_epochs: 
    n_iters: 
    ngpu: 
    snapshot_freq: 
    algo: 
    anneal_power: 
    normalize_energynet_input: 

  data:
    dataset: 
    motion_file:
  
  model:
    sigma_begin: 
    sigma_end:
    L: 
    in_dim: 

    numObsSteps:
    encoder_hidden_layers:
    latent_space_dim: 
    decoder_hidden_layers:
    
  optim:
    weight_decay: 
    optimizer: 
    lr: 
    beta1: 
    amsgrad: 

  # Change these to use the learnt energies to train a policy
  inference:
    task_reward_w: 
    energy_reward_w: 
    sigma_level: 
    eb_model_checkpoint: 
    running_mean_std_checkpoint: 
```