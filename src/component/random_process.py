#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np


class RandomProcess(object):
    def reset_states(self):
        pass


class GaussianProcess(RandomProcess):
    def __init__(self, size, std):
        self.size = size
        self.std = std

    def sample(self):
        return np.random.randn(*self.size) * self.std()


class OrnsteinUhlenbeckProcess(RandomProcess):
    """
    The Ornstein-Uhlenbeck process adds time-correlated noise to the actions taken by the deterministic policy.

    The OU process statisfied the following stochastic differential equation:
    dxt = theta * (mu - xt) * dt + sigma * dWt
    where Wt denotes the Wiener process

    The Wiener process
    f_Wt(x) = 1/sprt(2 * pi * t) exp(-x^2/2*t)
    Wt = Wt - W0 ~ N(0, t)
    """


    def __init__(self, size, std, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std() * np.sqrt(
            self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
