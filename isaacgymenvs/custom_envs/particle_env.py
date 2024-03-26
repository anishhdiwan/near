import gym
from gym import spaces

import collections
import numpy as np
import pygame

import shapely.geometry as sg
import cv2
import skimage.transform as st


def unnormalise_action(action, max_vel):
    """Unnormalise an input action from being in the range of Box([-1,-1], [1,1]) to the range Box([-max_vel,-max_vel], [max_vel, max_vel])

    Given,
    [r_min, r_max] = [-1,1] = data range
    [t_min, t_max] = [-max_vel, max_vel] = target range
    x in data range

    x_in_target_range = t_min + (x - r_min)*(t_max - t_min)/(r_max - r_min) 
    https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range

    Args:
        action (gym.Actions): Input action in normalised range
        max_vel (float): Max velocity to which the action is unnormalised
    """
    # action = (action + 1)*(max_vel)/2

    action = -max_vel + (action + 1)*(max_vel)

    return action

class Particle():
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius

class ParticleEnv(gym.Env):
    """
    A simple 2D environment with a particle that moves aroung in a window, given velocity actions

    render_action (Bool): Whether to render actions
    render_size (Int): Render scaling
    reset_to_state (gym.spaces.Box): Starting state on resetting the env
    cfg (dict config): Typically a hydra config with rendering settings or learning algorithm related params
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 5}
    reward_range = (0., 1.)

    def __init__(self,
            render_action=True,
            render_size=96,
            reset_to_state=None,
            cfg=None,
            normalise_action=True,
        ):
        self._seed = None
        self.normalise_action = normalise_action
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.max_vel = max_vel = 20 # px/s in each axis
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.control_hz = self.metadata['video.frames_per_second']

        # agent_pos
        self.observation_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        # Setting up the env observation space info used to train amp_continuous
        if cfg != None:
            # General config params
            self._headless = cfg["headless"]
            self._num_envs = cfg["env"]["numEnvs"]
            self._training_algo = cfg["training_algo"]

            try:
                # Training params for AMP and similar algos
                # self.reset_called = False

                # Number of features in an observation vector 
                NUM_OBS_PER_STEP = 2 # [robotY, robotY]
                self._num_obs_per_step = NUM_OBS_PER_STEP

                # Number of observations to group together. For example, AMP groups to observations s-s' together to compute rewards as discriminator(s,s')
                # numAMPObsSteps defines the number of obs to group in AMP. numObsSteps also defines the same but is a bit more general in its wording.
                # Support for numAMPObsSteps is kept to maintain compatibility with AMP configs
                try:
                    self._num_obs_steps = cfg["env"]["numAMPObsSteps"]
                except KeyError:
                    self._num_obs_steps = cfg["env"].get("numObsSteps", 2)

                self._motion_file = cfg["env"].get('motion_file', "random_motions.npy")
                
                assert(self._num_obs_steps >= 2)
                # Number of features in the grouped observation vector
                self.num_obs = self._num_obs_steps * NUM_OBS_PER_STEP

                # Observation space as a gym Box object
                self._obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
            except Exception as e:
                pass

        if self.normalise_action:
            # Normalised actions: in the range [-1,1]. Action space changed to normalised values
            # In the step() function, actions are unnormalised again
            self.action_space = spaces.Box(
                low=np.array([-1,-1], dtype=np.float64),
                high=np.array([1,1], dtype=np.float64),
                shape=(2,),
                dtype=np.float64
            )
        else:
            # velocity goal for agent
            self.action_space = spaces.Box(
                low=np.array([-max_vel,-max_vel], dtype=np.float64),
                high=np.array([max_vel,max_vel], dtype=np.float64),
                shape=(2,),
                dtype=np.float64
            )


        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.teleop = None
        self.latest_action = None
        self.reset_to_state = reset_to_state


    def seed(self, seed=None):
        """
        Set a seed or use a random seed
        """
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    # Added to provide env obs shape info to amp_continuous
    @property
    def paired_observation_space(self):
        return self._obs_space


    def _setup(self):
        self.teleop = False

        # Add agent, block, and goal zone.
        self.agent = Particle((256, 400), 15)
        self.goal_pose = np.array([self.window_size/2, self.window_size/2])

        # Counting the number of steps
        self.env_steps = 0
        # step() returns done after this
        self.max_env_steps = 150

    def reset(self):
        """
        Reset the environment 
        """
        seed = self._seed
        self._setup()
        
        state = self.reset_to_state
        if state is None:
            state = np.random.randint(low=50, high=450, size=2)
        self._set_state(state)

        observation = self._get_obs()
        return observation


    def reset_done(self):
        """
        Wrapper around reset to enable compatibility with the amp_continous play method
        """
        return self.reset(), []

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state
        self.agent.position = pos_agent



    def step(self, action):
        if self.normalise_action:
            # Unnormalise action before applying to the env
            action = unnormalise_action(action, self.max_vel)

        dt = 1.0 / self.sim_hz
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                updated_position = self.agent.position + action * dt

                if updated_position[0] < 0.:
                    updated_position[0] = 0.
                elif updated_position[0] > self.window_size:
                    updated_position[0] = self.window_size

                if updated_position[1] < 0.:
                    updated_position[1] = 0.
                elif updated_position[1] > self.window_size:
                    updated_position[1] = self.window_size

                self.agent.position = updated_position

                

        # TODO
        dist_to_goal = np.linalg.norm(np.absolute(self.agent.position) - np.absolute(self.goal_pose))/363
        reward = -dist_to_goal

        # TODO        
        done = dist_to_goal < 0.05
        
        
        observation = self._get_obs()
        info = self._get_info()

        # Rewards need to be in the info to get logged by the observer
        info['scores'] = reward

        #Env stops after a certain number of steps
        self.env_steps += 1
        if self.env_steps >= self.max_env_steps:
            done = True

        return observation, reward, done, info

    def render(self, mode):
        return self._render_frame(mode)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position))
        return obs


    def _get_info(self):
        info = {
            'pos_agent': np.array(self.agent.position)}
        return info


    def _render_frame(self, mode):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas


        # Draw agent (to self.screen)
        pygame.draw.circle(self.screen, pygame.Color('RoyalBlue'), self.agent.position, self.agent.radius)
        pygame.draw.circle(self.screen, pygame.Color('palegreen4'), self.goal_pose, 5)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"


        # img = np.transpose(
        #         np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        #     )
        # img = cv2.resize(img, (self.render_size, self.render_size))
        # if self.render_action:
        #     if self.render_action and (self.latest_action is not None):
        #         action = np.array(self.latest_action)
        #         coord = (action / 512 * 96).astype(np.int32)
        #         marker_size = int(8/96*self.render_size)
        #         thickness = int(1/96*self.render_size)
        #         cv2.drawMarker(img, coord,
        #             color=(255,0,0), markerType=cv2.MARKER_CROSS,
        #             markerSize=marker_size, thickness=thickness)
        # return img


    def teleop_agent(self):
        pass
        # TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        # def act(obs):
        #     act = None
        #     mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
        #     if self.teleop or (mouse_position - self.agent.position).length < 30:
        #         self.teleop = True
        #         act = mouse_position
        #     return act
        # return TeleopAgent(act)


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()