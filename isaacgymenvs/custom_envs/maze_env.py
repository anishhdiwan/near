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


def unnormalise_action_teleop(action, range_max):
    """Unnormalise an input action from being in the range of Box([-1,-1], [1,1]) to the range Box([-range_max,-range_max], [range_max, range_max])

    Given,
    [r_min, r_max] = [-1,1] = data range
    [t_min, t_max] = [-range_max, range_max] = target range
    x in data range

    x_in_target_range = t_min + (x - r_min)*(t_max - t_min)/(r_max - r_min) 
    https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range

    Args:
        action (gym.Actions): Input action in normalised range
        window_size (float): Size of the pushT window to which the action is unnormalised
    """

    action = range_max * action

    return action


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
    A simple 2D environment with a particle that moves given position actions. The environment also has a maze that offers a longer (suboptimal) path to the end. 
    
    During Learning: The action space is all possible points in the environment
    During Data Collection: The action space is the set of 2D poses in a k unit circle around the agent.

      .....
    .       .
    .   A    .
     .      .
      .....
    
    The spaces are different because data collection requires slower movements for easier control. Further, the max possible env steps is also higher during data collection
    for easier control. 


    render_action (Bool): Whether to render actions
    render_size (Int): Render scaling
    fixed_reset_state (gym.spaces.Box): Starting state on resetting the env
    cfg (dict config): Typically a hydra config with rendering settings or learning algorithm related params
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            render_action=True, damping=None,
            render_size=96,
            fixed_reset_state=None,
            cfg=None,
            normalise_action=True,
        ):
        self._seed = None
        self.normalise_action = normalise_action
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        # Radius of the circle around the agent in which position actions are given
        self.action_space_radius = act_rad = 3.0
        self.render_size = render_size
        self.sim_hz = 100
        # step() returns done after this
        self.max_env_steps = 650
        self.quit_if_stuck = True
        self.teleop = False

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
                low=np.array([-act_rad,-act_rad], dtype=np.float64),
                high=np.array([act_rad,act_rad], dtype=np.float64),
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
        shape = list(self.observation_space.shape)[0]
        self.last_states = []
        self.fixed_reset_state = fixed_reset_state


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
        # Set up a pumunk space for 2D physics
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
            self._add_polygon([(40,140), (40,460), (20,460), (20,140)]),
            self._add_polygon([(100,140), (100,380), (120,380), (120,140)]),
            self._add_polygon([(20,460), (350,460), (350,480), (20,480)]),
            self._add_polygon([(100,380), (350,380), (350,400), (100,400)]),
        ]
        self.space.add(*maze_walls)

        # https://htmlcolorcodes.com/color-names
        self.goal_color = pygame.Color("ForestGreen")
        self.goal_pose = 0.85 * np.array([self.window_size, self.window_size])

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)

        # Counting the number of steps
        self.env_steps = 0


    def _add_polygon(self, vertices):
        shape = pymunk.Poly(self.space.static_body, vertices)
        shape.color = pygame.Color("#B4656F")    # https://htmlcolorcodes.com/color-names #C33C54 #846C5B #B4656F
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
        self._setup()
        if self.damping is not None:
            self.space.damping = self.damping
        
        state = self.fixed_reset_state
        if state is None:
            state = self.np_random.integers(low=60, high=80, size=2)
        self._set_state(state)

        observation = self._get_obs()
        return observation


    def reset_to_state(self, state):
        """
        Reset the environment to a specified state
        """
        self._setup()
        if self.damping is not None:
            self.space.damping = self.damping
        
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
            if self.teleop:
                action = unnormalise_action_teleop(action, self.action_space_radius)
            else:
                # Unnormalise action before applying to the env
                action = unnormalise_action(action, self.window_size)

        if self.teleop:
            current_pos = np.array(tuple(self.agent.position))
            action = current_pos + action
            # # Time in seconds for which to apply an action
            action_time = 0.6
        else:
            # # Time in seconds for which to apply an action
            action_time = 0.3

        dt = action_time / self.sim_hz
        n_steps = self.sim_hz

        if action is not None:
            # Saving latest actions and states
            self.latest_action = action
            if len(self.last_states) > 100:
                self.last_states.pop(0)
                self.last_states.append(self.agent.position)
            else:
                self.last_states.append(self.agent.position)

            for i in range(n_steps):
                # self.agent.position = self.agent.position + action * dt
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                if not self.teleop:
                    acceleration = acceleration * 0.1
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)


        dist_to_goal = np.linalg.norm(np.absolute(np.array(self.agent.position)) - np.absolute(self.goal_pose))/np.linalg.norm(np.absolute(self.goal_pose))
        # reward = -dist_to_goal

        reward = 0.

        # done = dist_to_goal < 0.05
        done = False

        if dist_to_goal < 0.05:
            done = True
            reward += 1000

        if self.quit_if_stuck:
            if self.env_steps > 250:
                if self.is_stuck():
                    done = True
                    reward += -500
        
        
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

        # Draw the previous action
        if self.render_action and (self.latest_action is not None):
            action = np.array(self.latest_action)
            self.drawCrossHair(self.screen, action[0], action[1])

        # Draw all bodies defined in the space (agent, walls, maze)
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        save_canvas = False
        if save_canvas:
            if not hasattr(self, 'canvas_saved'):
                self.canvas_saved = False
            
            if not self.canvas_saved:
                pygame.image.save(self.window, "maze_env_screenshot.tga")
                self.canvas_saved=True



    def drawCrossHair(self, surface, x, y):
        size = 5
        pygame.draw.lines(surface, (255,0,0), True, [(x-size,y-size),(x+size,y+size)], 5)
        pygame.draw.lines(surface, (255,0,0), True, [(x-size,y+size),(x+size,y-size)], 5)

    def teleop_agent(self, record_data=True):
        self.quit_if_stuck = False
        self.teleop = True
        self.max_env_steps = 5000
        assert self.normalise_action == True, "Please set normalisation to True. This is necessary to feed in the correct joystick commands to the environment"
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
        # self.max_env_steps = 12000
        recording_stated = False

        # Reset env
        obs = self.reset()

        print("The environment will reset after a set time! Press Ctrl+C to force stop")
        done = False
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
                    states = ObservationArray(self.max_env_steps, self.action_space.shape[0])

                elif js_A and recording_stated==True:
                    recording_stated = True
                    print("Game reset during recording! Previous recording discarded and new one started")
                    time.sleep(1)
                    obs = self.reset()
                    states = ObservationArray(self.max_env_steps, self.action_space.shape[0])

                elif js_B and recording_stated==True:
                    recording_stated = False
                    print("Game reset during recording! Previous recording discarded")
                    time.sleep(1)
                    obs = self.reset()
            
            current_pos = np.array(tuple(self.agent.position))
            if record_data and recording_stated:
                # states.show_last()
                if self.env_steps % 10 == 0:
                    # states[self.env_steps] = current_pos
                    states.append(current_pos)

            action = np.array([js_x, js_y])
            # Only apply action if joystick vals are non-zero
            if not np.logical_and(action > -0.3, action < 0.3).all():
                observation, reward, done, info = self.step(action)

            # print(f"Obs {observation} | Rew {reward} | done {done} | info {info}")
            self.render(mode="human")

            if done:
                if record_data and recording_stated:
                    if not info.get('max_steps', False):
                        now = '_{date:%d-%H-%M-%S}'.format(date=datetime.now())
                        with open(f'data/maze_env/{now}.npy', 'wb') as f:
                            np.save(f, states.get_data())
                        print("Recording saved! Game reset to normal mode")
                    else:
                        print("Recording was not saved! Episode ended at max steps. Game reset")

                    recording_stated = False

                else:
                    print("Episode finished! Resetting")
                time.sleep(0.25)
                obs = self.reset()


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def is_stuck(self):
        first_state = self.last_states[0]
        last_state = self.last_states[-1]
        dist = np.linalg.norm(np.absolute(first_state) - np.absolute(last_state))/np.linalg.norm(np.array([self.window_size,self.window_size]))
        # If agent has not moved by some percentage of the window size then it is stuck
        if dist < 0.003:
            return True
        else:
            return False


class ObservationArray():

    def __init__(self, max_length, observation_dim):
        """A numpy-based array enabling appending

        Args:
            max_length (int): The maximum length of the array
            observation dim (int): The number of features in each observation
        """
        self.states = np.zeros((max_length, observation_dim))
        self.idx = 0

    def append(self, observation):
        """Append an observation to the array
        """
        self.states[self.idx] = observation
        self.idx += 1

    def get_data(self):
        """Return the filled up portion of the array
        """
        return self.states[:self.idx]

    def show_last(self):
        """Print the last 10 entries for debugging
        """
        print(self.states[self.idx - 10: self.idx])




if __name__ == "__main__":
    env = MazeEnv()
    env.teleop_agent(record_data=True)
