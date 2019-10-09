import os
import sys
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from collections import deque
import itertools
import time
import cv2


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


class FrankaReacherEnvDetect(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        action_range = np.linspace(-2.0, 2.0, 5)
        self.action_idx2real = list(itertools.product(action_range, repeat=7))
        self.count = 0

        self.high = np.array([40, 35, 30, 20, 15, 10, 10])
        self.low = -self.high
        self.wt = 0.0
        self.we = 0.0
        self.qlimit_min_hard = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973 ])
        self.qlimit_max_hard = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973 ])
        root_dir = os.path.dirname(__file__)
        xml_path = os.path.join(root_dir, 'single_franka', 'single_franka.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_path, 1)
        utils.EzPickle.__init__(self)

        # Manually define this to let a be in [-1, 1]^d
        self.action_space = spaces.Box(low=-np.ones(7) * 2, high=np.ones(7) * 2, dtype=np.float32)
        self.init_params()


    def init_params(self, wt=0.9, x=0.0, y=0.0, z=0.2):
        """
        :param wt: Float in range (0, 1), weight on euclidean loss
        :param x, y, z: Position of goal
        """
        self.wt = wt
        self.we = 1 - wt
        qpos = self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.1:
                break
        qpos[-3:-1] = self.goal
        qpos[-1] = z
        # qpos[-3:] = [x, y, z]
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

    def step(self, a):
        a_real = a * self.high / 2
        self.do_simulation(a_real, self.frame_skip)



        reward = self._reward(a_real)
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    def _reward(self, a):
        eef = self.get_body_com('panda_link7')
        goal = self.get_body_com('goal')
        goal_distance = np.linalg.norm(eef - goal)
        # This is the norm of the joint angles
        # The ** 4 is to create a "flat" region around [0, 0, 0, ...]
        # q_norm = np.linalg.norm(self.sim.data.qpos.flat[:7]) ** 4 / 100.0
        q_norm = 0

        # TODO in the future
        # f_desired = np.eye(3)
        # f_current = body_frame(self, 'gripper_r_base')

        reward = -(
            self.wt * goal_distance * 2.0 +  # Scalars here is to make this part of the reward approx. [0, 1]
            self.we * np.linalg.norm(a) / 40 +
            q_norm
        )
        return reward

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:7]
        image = self.render(mode='rgb_array', width=256, height=256 )
        """ object detector  """
        # Set up the detector with default parameters
        detector = cv2.SimpleBlobDetector_create()

        # Detect blobs
        keypoints = detector.detect(image)
        if keypoints:
            target_pos = keypoints[0].pt
            # print("Target: ", self.sim.data.qpos[-3:])
            # print("Detector: ", target_pos)
            # print("#" * 100)
            # sys.exit()

            mat_affine = np.array([[ 3.34066656e-03,  1.95891623e-04, -3.44271003e-01],
 [-2.67918668e-05, -5.44222813e-03,  4.73367672e-01],
 [-6.93889390e-18,  0.00000000e+00,  1.00000000e+00]], np.float64) # calculated manually, cf. affine_transformation.ipynb
            point_det = np.array([target_pos[0], target_pos[1], 1.0], np.float64).reshape(3,1)
            estimated_target_pos = np.matmul(mat_affine, point_det).reshape(-1,)
            # estimated_target_pos = estimated_target_pos[:-1]
            estimated_target_pos[-1] = self.get_body_com('goal')[-1]

        else:
            # if the object detector fails
            print("Object detector fails")
            estimated_target_pos = np.zeros(3)
            estimated_target_pos[-1] = self.get_body_com('goal')[-1]
        

        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            estimated_target_pos,
            self.sim.data.qvel[:7],
            self.get_body_com('panda_link7') - estimated_target_pos
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
        #self.init_qpos[:7] = np.random.uniform(pos_low, pos_high)
        init_q_r1=np.array([-0.26072199, -0.14981304, -0.46675336, -2.51181758, -0.09701893,  2.37547235,   0.13245205])
        #set inital pos and vel
        np.random.seed(int(time.time()))
        self.init_qpos[:7] = init_q_r1+np.random.uniform(-1,1,7)*0.1
        vel_high = np.ones(7) * 0.5
        vel_low = -vel_high
        self.init_qvel[:7] = np.random.uniform(vel_low, vel_high)
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.lookat[0] = 0.1
        self.viewer.cam.lookat[1] = -0.5
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.distance = 0.5
        self.viewer.cam.elevation = -45
        self.viewer.cam.azimuth = 90
        #
        # self.viewer.cam.lookat[0] = 0
        # self.viewer.cam.lookat[1] = 0
        # self.viewer.cam.lookat[2] = 0
        # self.viewer.cam.distance = 2.0
        # self.viewer.cam.elevation = -90
        # self.viewer.cam.azimuth = 90
