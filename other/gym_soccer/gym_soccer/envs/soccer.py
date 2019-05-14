import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces

class SoccerEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
      self.observation_space = spaces.Box(low=-1, high=1,shape=(3,3))
     
  def step(self, action):
    pass
  def reset(self):
    pass
  def render(self, mode='human'):
    pass
  def close(self):
    pass
