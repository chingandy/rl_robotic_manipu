#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import os
import gym
import numpy as np
import torch
from gym import wrappers
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import FrameStack as FrameStack_
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv
import time
from utils import *
import itertools
# import pyglet.window

# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(env_id, seed, rank, video_rendering, episode_life=True):
    def _thunk():
        random_seed(seed)
        if env_id.startswith("dm"):
            import dm_control2gym
            _, domain, task = env_id.split('-')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
            if video_rendering:
                env = wrappers.Monitor(env, './videos/' + str(time.localtime().tm_min) + '_' + str(time.localtime().tm_sec) + '/')

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        env = OriginalReturnWrapper(env)
        if is_atari:
            env = wrap_deepmind(env,
                                episode_life=episode_life,
                                clip_rewards=False,
                                frame_stack=False,
                                scale=False)
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3:
                env = TransposeImage(env)
            env = FrameStack(env, 4)

        return env

    return _thunk


class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


# The original LayzeFrames doesn't work well
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))




# class ActionWrapper(gym.ActionWrapper):
#     def __init__(self, env, dis_level):
#         super().__init__(env)
#         if dis_level == -1: # test mode, to test panda with only 2 dof
#             action_range = np.linspace(-2.0, 2.0, 11)
#             self.action_space = list(itertools.product(action_range, repeat=2))
#         elif dis_level:
#             # print("Action space: ", env.action_space.shape[0], type(env.action_space))
#             action_range = np.linspace(-2.0, 2.0, dis_level)
#             self.action_space = list(itertools.product(action_range, repeat=env.action_space.shape[0]))
#
#     def action(self, act):
#         # modify act
#         return act

# The original one in baselines is really bad
class DummyVecEnv(VecEnv):
    def __init__(self, env_fns, dis_level):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space) # adding the key 'episodic_return' because of VecEnv from the module "baselins"
        self.actions = None
        if dis_level == -1: # test mode, to test panda with only 2 dof
            action_range = np.linspace(-2.0, 2.0, 11)
            self.action_space = list(itertools.product(action_range, repeat=2))
        elif dis_level:
            # print("Action space: ", env.action_space.shape[0], type(env.action_space))
            action_range = np.linspace(-2.0, 2.0, dis_level)
            self.action_space = list(itertools.product(action_range, repeat=env.action_space.shape[0]))

        # if dis_level is not None:
        #     print("Discretization level: ", dis_level)
        #     action_range = np.linspace(-3.0, 3.0, dis_level)
        #     self.action_space = list(itertools.product(action_range, action_range))

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        # print("In step_wait")
        data = []
        for i in range(self.num_envs):
            # print("self.actions[i]  ",self.actions[i])
            obs, rew, done, info = self.envs[i].step(self.actions)
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info

    def reset(self):
        return [env.reset() for env in self.envs]

    def close(self):
        return


class Task:
    def __init__(self,
                 name,
                 video_rendering,
                 dis_level=None,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=np.random.randint(int(1e5))):
        if log_dir is not None:
            mkdir(log_dir)
        envs = [make_env(name, seed, i, video_rendering, episode_life) for i in range(num_envs)]
        if single_process:
            self.env = DummyVecEnv(envs, dis_level)
        else:
            self.env = SubprocVecEnv(envs)
        # if single_process:
        #     Wrapper = DummyVecEnv
        # else:
        #     Wrapper = SubprocVecEnv
        # self.env = Wrapper(envs)
        self.name = name
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        # self.action_dim = dis_level
        # if self.action_dim is None:
        #     print("Please specify the number of bins")
        #     quit()
        if dis_level is not None:
        # if name == "Reacher-v101" or name == "Reacher-v102":
            self.action_dim = len(self.action_space)
        elif isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)
