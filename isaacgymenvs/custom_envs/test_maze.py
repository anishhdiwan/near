from maze_env import MazeEnv
import time

env = MazeEnv(normalise_action=False)


env.teleop_agent(record_data=True)

# obs = env.reset()

# for _ in range(200):
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
#     # print(f"Obs {observation} | Rew {reward} | done {done} | info {info}")
#     env.render(mode="human")
#     time.sleep(0.1)


# import pygame
# import time
# # pygame.joystick.init()
# pygame.init()
# assert pygame.joystick.get_count() > 0, "No joystick found! Please plug in and try again"
# joystick = pygame.joystick.Joystick(0)

# num_axes = joystick.get_numaxes()
# while True:
#     for event in pygame.event.get(): # get the events (update the joystick)
#         if event.type == pygame.QUIT: # allow to click on the X button to close the window
#             pygame.quit()
#             exit()

#     joy_vals = {"x-axis":joystick.get_axis(0), "y-axis":joystick.get_axis(1), "A-button":joystick.get_button(0), "X-button":joystick.get_button(2)}
#     print(joy_vals)
