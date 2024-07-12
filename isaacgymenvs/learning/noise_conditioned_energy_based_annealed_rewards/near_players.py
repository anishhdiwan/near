import torch 
import numpy as np

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer
import gym
# from rl_games.common import env_configurations
from utils.ncsn_utils import dict2namespace
from learning.motion_ncsn.models.motion_scorenet import SimpleNet, SinusoidalPosEmb
import time
from isaacgymenvs.tasks.humanoid_amp import HumanoidAMP 

def get_augmented_env_info(env, **kwargs):
    if "temporal_feature" in list(kwargs.keys()):
        if kwargs["temporal_feature"] == True:
            assert "temporal_emb_dim" in list(kwargs.keys()), "A temporal embedding dim must be provided if encoding temporal features"
            temporal_emb_dim = kwargs["temporal_emb_dim"]

            result_shapes = {}
            result_shapes['action_space'] = env.action_space
            result_shapes['observation_space'] = gym.spaces.Box(np.ones(env.num_obs + temporal_emb_dim) * -np.Inf, np.ones(env.num_obs + temporal_emb_dim) * np.Inf)

            result_shapes['agents'] = 1
            result_shapes['value_size'] = 1
            if hasattr(env, "get_number_of_agents"):
                result_shapes['agents'] = env.get_number_of_agents()
            '''
            if isinstance(result_shapes['observation_space'], gym.spaces.dict.Dict):
                result_shapes['observation_space'] = observation_space['observations']
            
            if isinstance(result_shapes['observation_space'], dict):
                result_shapes['observation_space'] = observation_space['observations']
                result_shapes['state_space'] = observation_space['states']
            '''
            if hasattr(env, "value_size"):    
                result_shapes['value_size'] = env.value_size
            print(result_shapes)
            return result_shapes

    else:
        result_shapes = {}
        result_shapes['observation_space'] = env.observation_space
        result_shapes['action_space'] = env.action_space
        result_shapes['agents'] = 1
        result_shapes['value_size'] = 1
        if hasattr(env, "get_number_of_agents"):
            result_shapes['agents'] = env.get_number_of_agents()
        '''
        if isinstance(result_shapes['observation_space'], gym.spaces.dict.Dict):
            result_shapes['observation_space'] = observation_space['observations']
        
        if isinstance(result_shapes['observation_space'], dict):
            result_shapes['observation_space'] = observation_space['observations']
            result_shapes['state_space'] = observation_space['states']
        '''
        if hasattr(env, "value_size"):    
            result_shapes['value_size'] = env.value_size
        print(result_shapes)
        return result_shapes


class NEARPlayerContinuous(players.PpoPlayerContinuous):
    def __init__(self, params):
        """
        Initialise a player to run trained policies. Inherits from the PPO player and makes minor modifications to attach energy functions

        Args:
            params (:obj `dict`): Algorithm parameters (self.config is obtained from params in the parent class __init__() method)
        """

        config = params['config']

        # If using temporal feature in the state vector, create the environment first and then augment the env_info to account for extra dims
        if config['near_config']['model']['encode_temporal_feature']:
            print("Using Temporal Features")
            temporal_emb_dim = config['near_config']['model'].get('temporal_emb_dim', None)
            assert temporal_emb_dim != None, "A temporal embedding dim must be provided if encoding temporal features"
            self.env_name = config['env_name']
            self.env_config = config.get('env_config', {})
            env = self.create_env()
            self.env_info = get_augmented_env_info(env, temporal_feature=True, temporal_emb_dim=temporal_emb_dim)
            params['config']['env_info'] = self.env_info

        super().__init__(params)

        # Set the self.env attribute
        if config['near_config']['model']['encode_temporal_feature']:
            self.env = env

        self._task_reward_w = self.config['near_config']['inference']['task_reward_w']
        self._energy_reward_w = self.config['near_config']['inference']['energy_reward_w']
        self._eb_model_checkpoint = self.config['near_config']['inference']['eb_model_checkpoint']
        self._c = self.config['near_config']['inference']['sigma_level'] # c ranges from [0,L-1]
        self._L = self.config['near_config']['model']['L']
        self._normalize_energynet_input = self.config['near_config']['training'].get('normalize_energynet_input', True)
        self._energynet_input_norm_checkpoint = self.config['near_config']['inference']['running_mean_std_checkpoint']

        # If temporal features are encoded in the paired observations then a new space for the temporal states is made. The energy net and normalization use this space
        self._encode_temporal_feature = self.config['near_config']['model']['encode_temporal_feature']

        try:
            self._max_episode_length = self.env.max_episode_length
        except AttributeError as e:
            self._max_episode_length = None
        
        if self._encode_temporal_feature:
            assert self._max_episode_length != None, "A max episode length must be known when using temporal state features"

            # Positional embedding for temporal information
            self.emb_dim = config['near_config']['model']['temporal_emb_dim']
            self.embed = SinusoidalPosEmb(dim=self.emb_dim, steps=512)
            self.embed.eval()


        # self._init_network(self.config['near_config'])


    def _init_network(self, energynet_config):
        """Initialise the energy-based model based on the parameters in the config file

        Args:
            energynet_config (dict): Configuration parameters used to define the energy network
        """

        # Convert to Namespace() 
        energynet_config = dict2namespace(energynet_config)

        eb_model_states = torch.load(self._eb_model_checkpoint, map_location=self.device)
        energynet = SimpleNet(energynet_config).to(self.device)
        energynet = torch.nn.DataParallel(energynet)
        energynet.load_state_dict(eb_model_states[0])

        self._energynet = energynet
        self._energynet.eval()
        # self._sigmas = np.exp(np.linspace(np.log(self._sigma_begin), np.log(self._sigma_end), self._L))


    # def obs_to_torch(self, obs):
    #     obs = super().obs_to_torch(obs)
    #     obs_dict = {
    #         'obs': obs
    #     }
    #     return obs_dict


    def _env_reset_done(self):
        """Reset any environments that are in the done state. 
        
        Wrapper around the vec_env reset_done() method. Internally, it handles several cases of envs being done (all done, no done, some done etc.)
        """

        obs, done_env_ids = self.env.reset_done()
        return self.obs_to_torch(obs), done_env_ids


    def _post_step(self, info):
        """
        Process info dict to print results
        """
        self._print_debug_stats(info)
        return


    def _print_debug_stats(self, infos):
        """Print training stats for debugging. Usually called at the end of every training epoch

        Args:
            infos (dict): Dictionary containing infos passed to the algorithms after stepping the environment
        """

        # TODO: Handle the case where paired obs is not provided (isaacgym envs)
        paired_obs = infos['paired_obs']

        shape = list(paired_obs.shape)
        shape.insert(0,1)
        paired_obs = paired_obs.view(shape)
        energy_rew = self._calc_energy(paired_obs)

        print(f"Mean energy reward (across all envs): {energy_rew.mean()}")



    def _calc_energy(self, paired_obs):
        """Run the pre-trained energy-based model to compute rewards as energies

        Args:
            paired_obs (torch.Tensor): A pair of s-s' observations (usually extracted from the replay buffer)
        """

        ### TESTING - ONLY FOR PARTICLE ENV 2D ###
        # paired_obs = paired_obs[:,:,:2]
        ### TESTING - ONLY FOR PARTICLE ENV 2D ###
        paired_obs = self._preprocess_observations(paired_obs)
        # Reshape from being (horizon_len, num_envs, paired_obs_shape) to (-1, paired_obs_shape)
        original_shape = list(paired_obs.shape)
        paired_obs = paired_obs.reshape(-1, original_shape[-1])

        # Tensor of noise level to condition the energynet
        labels = torch.ones(paired_obs.shape[0], device=paired_obs.device) * self._c # c ranges from [0,L-1]
        
        with torch.no_grad():
            energy_rew = self._energynet(paired_obs, labels)
            original_shape[-1] = energy_rew.shape[-1]
            energy_rew = energy_rew.reshape(original_shape)

        return energy_rew

    def _preprocess_observations(self, obs):
        """Preprocess observations (normalization)

        Args:
            obs (torch.Tensor): observations to feed into the energy-based model
        """

        if self._normalize_energynet_input:
            obs = self._energynet_input_norm(obs)
        return obs

    # def get_action(self, obs_dict, is_determenistic = False):
    #     """Simple wrapper to pass in dict instead of array
    #     """
    #     output = super().get_action(obs_dict['obs'], is_determenistic)
    #     return output

    def run(self):
        """Run a set number of games by executing the learnt policy. Also print out cumulative rewards

        """
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        # is_deterministic = self.is_deterministic
        is_deterministic = True
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            self.env._state_init = HumanoidAMP.StateInit.Random
            obses = self.env.reset_all()['obs']
            batch_size = 1
            # batch_size = self.get_batch_size(obses['obs'], batch_size)
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                ## New Addition ##
                # TODO! reset_done() is a custom_vec_env method. It resets all done envs. The player class directly instantiates a single gym env
                # Hence reset_done() simply resets the env.
                # obses, done_env_ids = self._env_reset_done()

                # Append temporal feature to observations if needed
                if self._encode_temporal_feature:
                    progress0 = self.embed(self.env.progress_buf/self._max_episode_length)
                    obses = torch.cat((progress0, obses), -1)
                
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                obses, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1

                ## New Addition ##
                # TODO! Paired obs come from the custom_vec_env and not from the underlying gym env. The player class only instantiates a gym env
                # This means that there are no paired obs from which energies can be computed
                # self._post_step(info)

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)

                if done_count == 512:
                    games_played += 1
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:,
                                                          all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        cur_rewards_done = cur_rewards/done_count
                        cur_steps_done = cur_steps/done_count
                        if print_game_res:
                            print(f'reward: {cur_rewards_done:.1f} steps: {cur_steps_done:.1} w: {game_res:.1}')
                        else:
                            print(f'reward: {cur_rewards_done:.1f} steps: {cur_steps_done:.1f}')

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)