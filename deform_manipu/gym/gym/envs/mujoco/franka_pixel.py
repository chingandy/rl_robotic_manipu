import os
import sys
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


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


class FrankaReacherEnvPixel(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.count = 0
        self.high = np.array([40, 35, 30, 20, 15, 10, 10,10,10])
        self.low = -self.high
        self.wt = 0.0
        self.we = 0.0
        root_dir = os.path.dirname(__file__)
        xml_path = os.path.join(root_dir, 'franka', 'franka.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_path, 1)
        utils.EzPickle.__init__(self)

        # Manually define this to let a be in [-1, 1]^d
        self.action_space = spaces.Box(low=-np.ones(9) * 2, high=np.ones(9) * 2, dtype=np.float32)
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
        vec = self.get_body_com("panda_leftfinger")-self.get_body_com("goal")
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


    def _get_obs(self):

        image = self.render(mode='rgb_array', width=256, height=256 )
        img_top, img_side = self.env.multi_render()
        print("img_top: ", type(img_top))        
        return image


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
        self.viewer.cam.lookat[2] = 0.4
        self.viewer.cam.distance = 2.0
        self.viewer.cam.elevation = -25
        self.viewer.cam.azimuth = 135
        # self.viewer.cam.lookat[0] = 0
        # self.viewer.cam.lookat[1] = 0
        # self.viewer.cam.lookat[2] = 0
        # self.viewer.cam.distance = 2.0
        # self.viewer.cam.elevation = -90
        # self.viewer.cam.azimuth = 90
