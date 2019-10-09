import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from collections import deque
import matplotlib.pyplot as plt
import cv2
import itertools

class ReacherEnvDetect(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.count = 0
        self.rewards = deque(maxlen = 3)
        self.high = np.array([3.0, 3.0])
        self.lazy = deque(maxlen = 3)
        utils.EzPickle.__init__(self) # some constructor
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
        # action_range = [-0.05, 0.05]
        # action_range = np.linspace(-3.0, 3.0, 15)
        # self.action_space = list(itertools.product(action_range, action_range))
        #action_range = np.arange(-np.pi, np.pi, 2 * np.pi / 360)
        # self.action_space = [[x, 0] for x in action_range]
        # self.action_space = [[x, 0] for x in action_range] + [[0, x] for x in action_range]



    def step(self, a):
        a = a * self.high / 2.0
        flag = 0  # 0: original, 1: simple reward, 2: intermidiate rewards
        # print("in step action: ", a)
        if flag == 0:
            """ reward function 0 """
            self.count += 1
            vec = self._get_obs()[-3:]
            # vec = self.get_body_com("fingertip")-self.get_body_com("target")
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
            # previous_vec = self.get_body_com("fingertip")-self.get_body_com("target")
            previous_vec = self._get_obs()[-3:]
            prev_dis = np.linalg.norm(previous_vec)
            self.do_simulation(a, self.frame_skip)
            ob = self._get_obs()
            vec = obs[-3:]
            # vec = self.get_body_com("fingertip")-self.get_body_com("target")
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
            # vec = self.get_body_com("fingertip")-self.get_body_com("target")
            vec = obs[-3:]
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

        # vec = self._get_obs()[-3:]
        # # vec = self.get_body_com("fingertip")-self.get_body_com("target")
        # reward_dist = - np.linalg.norm(vec)
        # reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + reward_ctrl
        # self.do_simulation(a, self.frame_skip)
        # ob = self._get_obs()
        # done = False
        # return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0 # 0
        self.viewer.cam.lookat[0] = 0 # added by andy
        self.viewer.cam.lookat[1] = 0 # added by andy
        self.viewer.cam.lookat[2] = 0 # added by andy
        self.viewer.cam.distance = 0.8
        self.viewer.cam.elevation = -90  # this denotes the direction of the viewer/ added by andy

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
        image = self.render(mode='rgb_array', width=256, height=256 ) # added by Andy, type: numpy.ndarray

        """ object detector  """
        # Set up the detector with default parameters
        detector = cv2.SimpleBlobDetector_create()

        # Detect blobs
        keypoints = detector.detect(image)
        if keypoints:
            target_pos = keypoints[0].pt
            # print("Detector: ", target_pos)


            mat_affine = np.array([[ 2.55730197e-03, 2.38344652e-06, -3.26386272e-01],
                         [-2.95326462e-06, -2.55334055e-03,  3.25817523e-01],
                         [-1.73472348e-18, -3.46944695e-18,  1.00000000e+00]], np.float64) # calculated manually, cf. affine_transformation.ipynb
            point_det = np.array([target_pos[0], target_pos[1], 1.0], np.float64).reshape(3,1)
            estimated_target_pos = np.matmul(mat_affine, point_det).reshape(-1,)
            estimated_target_pos[-1] = self.get_body_com("fingertip")[-1]
            # estimated_target_pos = estimated_target_pos[:-1]

        else:
            # if the object detector fails
            estimated_target_pos = np.zeros((3,))
            estimated_target_pos[-1] = self.get_body_com("fingertip")[-1]

        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            estimated_target_pos[:-1],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - estimated_target_pos
        ])
