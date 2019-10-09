from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco.humanoid import HumanoidEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.envs.mujoco.reacher import ReacherEnv
from gym.envs.mujoco.swimmer import SwimmerEnv
from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from gym.envs.mujoco.pusher import PusherEnv
from gym.envs.mujoco.thrower import ThrowerEnv
from gym.envs.mujoco.striker import StrikerEnv
from gym.envs.mujoco.yumi import YumiReacherEnv
from gym.envs.mujoco.franka import FrankaReacherEnv
from gym.envs.mujoco.franka_single import SingleFranka
from gym.envs.mujoco.franka_detect import FrankaReacherEnvDetect
from gym.envs.mujoco.franka_pixel import FrankaReacherEnvPixel
from gym.envs.mujoco.reacher_pixel import ReacherEnvPixel
from gym.envs.mujoco.reacher_detect import ReacherEnvDetect
