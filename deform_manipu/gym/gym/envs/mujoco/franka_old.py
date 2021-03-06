import os
import sys
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from collections import deque
import itertools


def body_index(model, body_name):
    return model.body_names.index(body_name)


def body_pos(model, body_name):
    ind = body_index(model, body_name)
    return model.body_pos[ind]


def body_quat(model, body_name):
    ind = body_index(model, body_name)
    return model.body_quat[ind]


def body_frame(env, body_name):
    """
    Returns the rotation matrix to convert to the frame of the named body
    """
    ind = body_index(env.model, body_name)
    b = env.data.body_xpos[ind]
    q = env.data.body_xquat[ind]
    qr, qi, qj, qk = q
    s = np.square(q).sum()
    R = np.array([
        [1 - 2 * s * (qj ** 2 + qk ** 2), 2 * s * (qi * qj - qk * qr), 2 * s * (qi * qk + qj * qr)],
        [2 * s * (qi * qj + qk * qr), 1 - 2 * s * (qi ** 2 + qk ** 2), 2 * s * (qj * qk - qi * qr)],
        [2 * s * (qi * qk - qj * qr), 2 * s * (qj * qk + qi * qr), 1 - 2 * s * (qi ** 2 + qj ** 2)]
    ])
    return R


class FrankaReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        action_range = np.linspace(-2.0, 2.0, 5)
        self.action_idx2real = list(itertools.product(action_range, repeat=7))
        self.count = 0
        self.rewards = deque(maxlen = 10)
        self.high = np.array([40, 35, 30, 20, 15, 10, 10])
        self.low = -self.high
        self.wt = 0.0
        self.we = 0.0
        root_dir = os.path.dirname(__file__)
        xml_path = os.path.join(root_dir, 'franka', 'franka.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_path, 1)
        utils.EzPickle.__init__(self)

        # Manually define this to let a be in [-1, 1]^d
        self.action_space = spaces.Box(low=-np.ones(7) * 2, high=np.ones(7) * 2, dtype=np.float32)
        # self.action_space = spaces.Box(low=-np.ones(9) * 2, high=np.ones(9) * 2, dtype=np.float32)


        self.init_params()

    def init_params(self, wt=0.9, x=0.0, y=0.0, z=0.2):
        """
        :param wt: Float in range (0, 1), weight on euclidean loss
        :param x, y, z: Position of goal
        """
        self.wt = wt
        self.we = 1 - wt
        qpos = self.init_qpos
        qpos[-3:] = [x, y, z]
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

    def step(self, a):
        self.count += 1
        # print("In Franka env, action: ", a, type(a))
        if self.count > 3 and (not isinstance(a, list)):
            a = self.action_idx2real[a.item()]
        # print("after:  ", a)
        a_real = a * self.high / 2
        """ Reward function 1 """

        vec = self.get_body_com("panda_leftfinger")-self.get_body_com("goal")
        done = False
        # print("Distance:", np.linalg.norm(vec), "  when vec: ", vec)
        if np.linalg.norm(vec) <= 0.0001 and self.count > 2:
            done = True
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a_real).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a_real, self.frame_skip)
        ob = self._get_obs()
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

        """ Reward function 2 """
        # self.do_simulation(a, self.frame_skip)
        # ob = self._get_obs()
        # vec = self.get_body_com("panda_leftfinger")-self.get_body_com("goal")
        # dis = np.linalg.norm(vec)
        # gamma = 0.25
        # done = False
        # if dis < 0.001:
        #     reward = 10
        #     done = True
        #     self.rewards.append(reward)
        #     return ob, reward, done, dict(rewards=self.rewards)
        # else:
        #     reward = np.exp(-gamma * dis)
        #     self.rewards.append(reward)
        #
        # return ob, reward, done, dict(rewards=self.rewards)


    # def _reward(self, a):
    #
    #     vec = self.get_body_com("panda_leftfinger")-self.get_body_com("goal")
    #     reward_dist = - np.linalg.norm(vec)
    #     reward_ctrl = - np.square(a).sum()
    #     reward = reward_dist + reward_ctrl
    #     self.do_simulation(a, self.frame_skip)
    #     ob = self._get_obs()
    #     done = False
    #     return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)


        """ original """
        # eef = self.get_body_com('panda_leftfinger')
        # goal = self.get_body_com('goal')
        # goal_distance = np.linalg.norm(eef - goal)
        # # This is the norm of the joint angles
        # # The ** 4 is to create a "flat" region around [0, 0, 0, ...]
        # q_norm = np.linalg.norm(self.sim.data.qpos.flat[:7]) ** 4 / 100.0
        #
        # # TODO in the future
        # # f_desired = np.eye(3)
        # # f_current = body_frame(self, 'gripper_r_base')
        #
        # reward = -(
        #     self.wt * goal_distance * 2.0 +  # Scalars here is to make this part of the reward approx. [0, 1]
        #     self.we * np.linalg.norm(a) / 40 +
        #     q_norm
        # )
        # return reward

    def _get_obs(self):
        # qpos =  self.sim.data.qpos
        # qpos_flat = self.sim.data.qpos.flat
        #
        # qvel =  self.sim.data.qvel
        # qvel_flat =  self.sim.data.qvel.flat
        # print("qpos: ", type(qpos), qpos.shape)
        #
        # print("qvel: ", type(qvel), qvel.shape)
        theta = self.sim.data.qpos.flat[:7]
        # obs = np.concatenate([
        #     np.cos(theta),
        #     np.sin(theta),
        #     self.sim.data.qpos[10:],
        #     self.sim.data.qvel[:10],
        #     self.get_body_com('panda_leftfinger') - self.get_body_com('goal')
        # ]
        # )
        # print(obs.shape)
        # print(obs)
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos[7:],
            self.sim.data.qvel[:9],
            self.get_body_com('panda_leftfinger') - self.get_body_com('goal')
        ]
        )
        # print("panda_leftfinger: ",self.get_body_com('panda_leftfinger'))
        # print("panda_rightfinger: ",self.get_body_com('panda_rightfinger'))

        """ original """
        # return np.concatenate([
        #     self.sim.data.qpos.flat[:7],
        #     np.clip(self.sim.data.qvel.flat[:7], -10, 10)
        # ])

    def reset_model(self):
        #pos_low  = np.array([-1.0,-0.3,-0.4,-0.4,-0.3,-0.3,-0.3])
        #pos_high = np.array([ 0.4, 0.6, 0.4, 0.4, 0.3, 0.3, 0.3])
        #self.init_qpos[:9] = np.random.uniform(pos_low, pos_high)*0
        self.init_qpos[:9] = [-0.37717236410840405, 0.07726983970821949, 0.4967134723412363, -1.6799738945375406, 0.18685498124837635, 3.2788068058225837, 2.149522179528243, 0.0399184376001358, 0.0399184376001358]
        #vel_high = np.ones(9) * 0.5
        #vel_low = -vel_high
        #self.init_qvel[:9] = np.random.uniform(vel_low, vel_high)*0
        self.init_qvel[:9]= [0,0,0,0,0,0,0,0,0]
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0.3
        self.viewer.cam.distance = 2.0
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 135
