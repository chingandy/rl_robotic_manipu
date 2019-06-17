import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from collections import deque

class ReacherEnvCNN(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.rewards = deque(maxlen = 3)
        utils.EzPickle.__init__(self) # some constructor
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
        # self.action_space = [[-0.0001, 0],[-0.01, 0],[0.0001, 0], [0.01, 0],[0 , -0.0001], [0, -0.01],[0, 0.0001], [0, 0.01] ]  # self-added
        # action_range = [-0.05, -0.0001, 0.0001, 0.05]
        action_range = [-0.0005, 0.0005]
        self.action_space = [[x, 0] for x in action_range] + [[0, x] for x in action_range] + [[0, 0]]
        # add_range = [-0.5, 0.5]
        # self.action_space = [[x, 0] for x in action_range] + [[0, x] for x in action_range] + [[y, 0] for y in add_range] + [[0, y] for y in add_range]


    def step(self, a):
        # a = self.action_space[a] # self-added
        # print("In step ation: ", a)
        ###################################
        # the reward function from https://arxiv.org/pdf/1511.03791.pdf
        previous_vec = self.get_body_com("fingertip")-self.get_body_com("target")
        prev_dis = np.linalg.norm(previous_vec)
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        dis = np.linalg.norm(vec)
        delta_dis = dis - prev_dis
        touch = False # specify if the target is touched or not
        gamma = 0.25


        """ Reward function 1 """
        # if delta_dis > 0:
        #     reward = -1
        # elif dis < 0.001:
        #     reward = 100
        #     done = True
        #     return ob, reward, done, info
        # elif delta_dis < 0:
        #     reward = 1
        # else:
        #     reward = 0
        # self.rewards.append(reward)
        # # print("######rewards log: ", self.rewards)
        # done = False
        # if len(self.rewards) < 3:
        #     pass
        # elif sum(self.rewards) < -1:
        #     done = True
        # else:
        #     done = False
        """ Reward function 2 """
        # if dis < 0.02:
        #     print("@"*10)
        #     print("Distance: ", dis)
        if delta_dis > 0:
            reward = - np.exp(gamma * dis)
        elif dis < 0.001:
            reward = 100
            touch = True
            done = True
            return ob, reward, done, touch
        elif delta_dis < 0:
            reward = np.exp(-gamma * dis)
        else:
            reward = 0

        self.rewards.append(reward)
        # print("######rewards log: ", self.rewards)
        done = False
        if len(self.rewards) < 3:
            pass
        elif sum(self.rewards) < 0:
            done = True
        else:
            done = False


        ###################################
        # vec = self.get_body_com("fingertip")-self.get_body_com("target")
        # reward_dist = - np.linalg.norm(vec)
        # reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + reward_ctrl
        # self.do_simulation(a, self.frame_skip)
        # ob = self._get_obs()
        # done = False
        ###################################
        # if done:
        #     print("In def step.......", done)
        # info = "this is from my step" + str(done)
        return ob, reward, done, touch
        ######################################
        # return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0 # 0
        self.viewer.cam.lookat[0] = 0 # added by andy
        self.viewer.cam.lookat[1] = 0 # added by andy
        self.viewer.cam.lookat[2] = 0 # added by andy
        self.viewer.cam.distance = 0.8
        self.viewer.cam.elevation = -90  # this denotes the direction of the viewer/ added by andy
        # print("#######",self.viewer.cam.lookat)
        # print(self.viewer.cam)

        # print(type(self.viewer.cam.lookat))
    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        # out = np.concatenate([
        #     np.cos(theta),
        #     np.sin(theta),
        #     self.sim.data.qpos.flat[2:],
        #     self.sim.data.qvel.flat[:2],
        #     self.get_body_com("fingertip") - self.get_body_com("target")
        # ])
        # print("out: ", out.shape)
        # print("cos(theta): ", np.cos(theta))
        # print("sin(theta): ", np.sin(theta))
        # print("qpos: ", self.sim.data.qpos.flat[2:])
        # print("qvel: ", self.sim.data.qvel.flat[:2])
        # print("fingertip: ", self.get_body_com("fingertip"))
        # print("target:", self.get_body_com("target"))
        # print(out)
        image = self.render(mode='rgb_array', width=256, height=256 ) # added by Andy, type: numpy.ndarray
        # return np.concatenate([
        #     np.cos(theta),
        #     np.sin(theta),
        #     self.sim.data.qpos.flat[2:],
        #     self.sim.data.qvel.flat[:2],
        #     self.get_body_com("fingertip") - self.get_body_com("target")
        # ])
        return image
