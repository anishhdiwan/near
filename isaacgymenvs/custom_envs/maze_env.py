import gym
from gym import spaces

import collections
import numpy as np
import pygame
import time
from datetime import datetime

import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st

# Importing from the file path
import sys
import os
PYMUNK_OVERRIDE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PYMUNK_OVERRIDE_PATH)
from pymunk_override import DrawOptions


def unnormalise_action(action, window_size):
    """Unnormalise an input action from being in the range of Box([-1,-1], [1,1]) to the range Box([0,0], [window_size, window_size])

    Given,
    [r_min, r_max] = [-1,1] = data range
    [t_min, t_max] = [0, window_size] = target range
    x in data range

    x_in_target_range = t_min + (x - r_min)*(t_max - t_min)/(r_max - r_min) 
    https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range

    Args:
        action (gym.Actions): Input action in normalised range
        window_size (float): Size of the pushT window to which the action is unnormalised
    """
    action = (action + 1)*(window_size)/2

    return action


class MazeEnv(gym.Env):
    """
    A simple 2D environment with a particle that moves given position actions. The goal is to get to the end.
    
    The environment also has a maze that offers a longer (suboptimal) path to the end. 

    render_action (Bool): Whether to render actions
    render_size (Int): Render scaling
    reset_to_state (gym.spaces.Box): Starting state on resetting the env
    cfg (dict config): Typically a hydra config with rendering settings or learning algorithm related params
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            render_action=True, damping=None,
            render_size=96,
            reset_to_state=None,
            cfg=None,
            normalise_action=True,
        ):
        self._seed = None
        self.normalise_action = normalise_action
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        # self.max_vel = max_vel = 2 # px/s in each axis
        self.render_size = render_size
        self.sim_hz = 100
        # step() returns done after this
        self.max_env_steps = 150

        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
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
                NUM_OBS_PER_STEP = 2 # [robotX, robotY]
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
                low=np.array([0,0], dtype=np.float64),
                high=np.array([ws,ws], dtype=np.float64),
                shape=(2,),
                dtype=np.float64
            )

        self.damping = damping
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
        self.pygame_initialised = False

        # Pumunk space
        self.space = None
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
        # Set up a pumunk space of 2D physics
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0


        # Add walls so that the agent cant not escape the environment
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, maze, and goal zone (goal pose is at k*ws in each axis where k in (0,1) is some const.)
        self.agent = self.add_circle((256, 400), 15)

        # Add maze
        maze_walls = [
            self._add_polygon([(40,100), (40,460), (20,460), (20,100)]),
            self._add_polygon([(100,100), (100,380), (120,380), (120,100)]),
            self._add_polygon([(20,460), (380,460), (380,480), (20,480)]),
            self._add_polygon([(100,380), (380,380), (380,400), (100,400)]),
        ]
        self.space.add(*maze_walls)

        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = 0.85 * np.array([self.window_size, self.window_size])

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        # self.collision_handeler.post_solve = self._handle_collision
        # self.n_contact_points = 0

        # Counting the number of steps
        self.env_steps = 0


    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)


    def _add_polygon(self, vertices):
        shape = pymunk.Poly(self.space.static_body, vertices)
        shape.color = pygame.Color('RosyBrown')    # https://htmlcolorcodes.com/color-names
        return shape


    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape


    def add_circle(self, position, radius):
        """Add a circle to the pymunk space
        """
        body = pymunk.Body(1, float("inf"))
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def reset(self):
        """
        Reset the environment 
        """
        seed = self._seed
        self._setup()
        if self.damping is not None:
            self.space.damping = self.damping
        
        state = self.reset_to_state
        if state is None:
            state = np.random.randint(low=60, high=80, size=2)
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

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)


    def step(self, action):
        if self.normalise_action:
            # Unnormalise action before applying to the env
            action = unnormalise_action(action, self.window_size)

        dt = 1.0 / self.sim_hz
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # self.agent.position = self.agent.position + action * dt
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)


        dist_to_goal = np.linalg.norm(np.absolute(self.agent.position) - np.absolute(self.goal_pose))/np.linalg.norm(np.absolute(self.goal_pose))
        reward = -dist_to_goal

        done = dist_to_goal < 0.05
        
        
        observation = self._get_obs()
        info = self._get_info()

        # Rewards need to be in the info to get logged by the observer
        info['scores'] = reward

        #Env stops after a certain number of steps
        self.env_steps += 1
        if self.env_steps >= self.max_env_steps:
            done = True
            info['max_steps'] = True

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
            if not self.pygame_initialised:
                pygame.init()
                # pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas
        
        draw_options = DrawOptions(canvas)
        # draw_options = pymunk.pygame_util.DrawOptions(canvas)

        # Draw goal pose
        pygame.draw.circle(self.screen, self.goal_color, self.goal_pose, 10)

        # Draw all bodies defined in the space (agent, walls, maze)
        self.space.debug_draw(draw_options)


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


    def teleop_agent(self, record_data=True):
        assert self.normalise_action == False, "Please set normalisation to False. This is necessary to feed in the correct joystick commands to the environment"
        # Get pygame joystick

        # Print instructions
        print('''At any point the left joystick moves the agent 
        
        Press B to reset the environment
        Press X to close the game
        Press A to reset the environment and start recording motion data (once recording has finished the game will resume in non-recording mode)
            - If A is pressed during recording, the current one is discarded and a new recording is started
            - If B is pressed during recording, the current one is discarded and the game resumes in normal mode 
        
        ''')

        if not self.pygame_initialised:
            pygame.init()
            # pygame.display.init()

        assert pygame.joystick.get_count() > 0, "No joystick found! Please plug in and try again"
        joystick = pygame.joystick.Joystick(0)

        # Increase max env steps before reset for teleop
        self.max_env_steps = 15000
        recording_stated = False

        # Reset env
        obs = self.reset()

        print("The environment will reset after a set time! Press Ctrl+C to force stop")
        while True:
            # Get joystick vals
            # Axes are in a range of [-1,1]
            js_x = joystick.get_axis(0)
            js_y = joystick.get_axis(1)

            # Buttons are bools
            js_A = joystick.get_button(0)
            js_B = joystick.get_button(1)
            js_X = joystick.get_button(2)


            # Quit if ctrl+c or window closed
            for event in pygame.event.get(): # get the events (update the joystick)
                if event.type == pygame.QUIT: # allow to click on the X button to close the window
                    pygame.quit()
                    exit()
                    
            # Quit if X
            if js_X:
                print("Quitting the game!")
                time.sleep(0.25)
                pygame.quit()
                exit()

            # Reset if B
            if not recording_stated:
                if js_B:
                    print("Resetting the game!")
                    time.sleep(0.25)
                    obs = self.reset()

            
            # Reset and start recording if A
            if record_data:
                if js_A and recording_stated==False:
                    recording_stated = True
                    print("Resetting the game! This episode will be recorded as training data")
                    time.sleep(0.5)
                    obs = self.reset()
                    states = np.zeros((self.max_env_steps, self.action_space.shape[0]))

                elif js_A and recording_stated==True:
                    recording_stated = True
                    print("Game reset during recording! Previous recording discarded and new one started")
                    time.sleep(1)
                    obs = self.reset()
                    states = np.zeros((self.max_env_steps, self.action_space.shape[0]))

                elif js_B and recording_stated==True:
                    recording_stated = False
                    print("Game reset during recording! Previous recording discarded")
                    time.sleep(1)
                    obs = self.reset()
            
            current_pos = np.array(tuple(self.agent.position))
            if record_data and recording_stated:
                states[self.env_steps] = current_pos

            js_action = self.scale_joystick(np.array([js_x, js_y]))
            action = current_pos + js_action
            observation, reward, done, info = self.step(action)

            # print(f"Obs {observation} | Rew {reward} | done {done} | info {info}")
            self.render(mode="human")

            if done:
                if record_data and recording_stated:
                    if not info.get('max_steps', False):
                        now = '_{date:%d-%H-%M-%S}'.format(date=datetime.now())
                        with open(f'data/maze_env/{now}.npy', 'wb') as f:
                            np.save(f, states)
                        print("Recording saved! Game reset to normal mode")
                    else:
                        print("Recording was not saved! Episode ended at max steps. Game reset")

                    recording_stated = False

                else:
                    print("Episode finished! Resetting")
                time.sleep(0.25)
                obs = self.reset()



    def scale_joystick(self, joystick_vals):
        """
        Scale joystick actions to increase or decrease input sensitivity
        """
        return 2.0 * joystick_vals


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()