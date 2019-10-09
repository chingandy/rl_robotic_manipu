import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from collections import deque

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.count = 0
        self.high = np.array([3.0, 3.0])
        self.rewards = deque(maxlen = 3) # self-added
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)



    def step(self, a):
        a = a * self.high / 2.0
        flag = 0  # 0: original, 1: simple reward, 2: intermidiate rewards
        # print("in step action: ", a)
        if flag == 0:
            """ reward function 0 """
            self.count += 1
            vec = self.get_body_com("fingertip")-self.get_body_com("target")
            done = False
            # print("Distance:", np.linalg.norm(vec), "  when vec: ", vec)
            if np.linalg.norm(vec) <= 0.0001 and self.count > 2:
                done = True
            reward_dist = - np.linalg.norm(vec)
            reward_ctrl = - np.square(a).sum()
            reward = reward_dist + reward_ctrl
            self.do_simulation(a, self.frame_skip)
            ob = self._get_obs()
            return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        elif flag == 1:
            """ reward function 1 """
            previous_vec = self.get_body_com("fingertip")-self.get_body_com("target")
            prev_dis = np.linalg.norm(previous_vec)
            self.do_simulation(a, self.frame_skip)
            ob = self._get_obs()
            vec = self.get_body_com("fingertip")-self.get_body_com("target")
            dis = np.linalg.norm(vec)
            delta_dis = dis - prev_dis
            gamma = 0.25
            done = False

            if delta_dis > 0:
                reward = -1
            elif dis < 0.01:
                reward = 100
                done = True
                self.rewards.append(reward)
                return ob, reward, done,  dict(rewards=self.rewards)
            elif delta_dis < 0:
                reward = 1
            else:
                reward = 0
            self.rewards.append(reward)

            if len(self.rewards) < 3:
                pass
            elif sum(self.rewards) < -1:
                done = True
            else:
                done = False
            return ob, reward, done, dict(rewards=self.rewards)
        elif flag == 2:
            self.do_simulation(a, self.frame_skip)
            ob = self._get_obs()
            vec = self.get_body_com("fingertip")-self.get_body_com("target")
            dis = np.linalg.norm(vec)
            gamma = 0.25
            done = False
            if dis < 0.001:
                reward = 10 + np.exp(-gamma * dis)
                done = True
                self.rewards.append(reward)
                return ob, reward, done, dict(rewards=self.rewards)
            else:
                reward = np.exp(-gamma * dis)
                self.rewards.append(reward)

            return ob, reward, done, dict(rewards=self.rewards)


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

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
        
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
