import gym
import time
import numpy as np


""" test the range of the two joints
Result
joint 0: no limit
joint 1: -3.0 ~ 3.0

 """
# env = gym.make("Reacher-v2")
# state = env.reset()
# # print out the action space
# # print( env.sim.data.qpos.flat[:2])
# step = 0
# for _ in range(100):
#     step += 1
#     print("Step: ", step)
# #     action = random.choice(env.action_space)
#     # action = env.action_space[0]
#     action = np.array([0, -1]) #* np.pi / 180.0 # degree to radian
#     print("Action: ", action)
#     env.step(action)
#     env.render()
# time.sleep(2)
# env.close()

"""
Create the action range with numpy.arange
"""

x = np.arange(-np.pi, np.pi, 2 * np.pi / 360)
print(x)
