import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
        """ Enable this block if want to train in the discrete action space """
        # action_range = [-0.1, -0.01, -0.005, 0.005, 0.01, 0.1]
        # action_range = [0.05, -0.05]
        # self.action_space = [[0, x] for x in action_range] + [[x, 0] for x in action_range] + [[0, 0]]


    def step(self, a):
        """ Original reward fucntion """
        ################################################################

        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        # return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        ####################################################################
        # previous_vec = self.get_body_com("fingertip")-self.get_body_com("target")
        # prev_dis = np.linalg.norm(previous_vec)
        # self.do_simulation(a, self.frame_skip)
        # ob = self._get_obs()
        # vec = self.get_body_com("fingertip")-self.get_body_com("target")
        # dis = np.linalg.norm(vec)
        # delta_dis = dis - prev_dis
        # gamma = 0.25
        # touch = False
        # done = False
        #
        # # test new constraints
        # vec_body_1 = self.get_body_com("body1") - self.get_body_com("world")
        # vec_target = self.get_body_com("target") - self.get_body_com("world")
        # inner_prod = np.dot(vec_body_1, vec_target)
        # # print("#" * 50)
        # # print("vec_body_1: ", vec_body_1)
        # # print("vec_target: ", vec_target)
        # # print("Inner product: ", np.dot(vec_body_1, vec_target))


        # image = self.render(mode='rgb_array', width=256, height=256 ) # added by Andy, type: numpy.ndarray

        """ Reward function 1 """
        # if delta_dis > 0:
        #     reward = -1
        # # elif dis < 0.01:
        # #     image = self.render(mode='rgb_array', width=256, height=256 )
        # #     plt.axis('off')
        # #     plt.imshow(image)
        # #     plt.savefig('touch'+ str(dis)[:5]+ '.png',transparent = True, bbox_inches = 'tight', pad_inches = 0)
        # #     print("##########################Image saved.###########################")
        # #     reward = 100
        # #     done = True
        # #     touch = True
        # #     return ob, reward, done, touch
        # elif delta_dis < 0:
        #     reward = 1
        # else:
        #     reward = 0
        # if inner_prod < 0:
        #     reward += 1000 * inner_prod
        # self.rewards.append(reward)
        # # print("######rewards log: ", self.rewards)
        #
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
        # if delta_dis > 0:
        #     reward = - np.exp(gamma * dis)
        # elif dis < 0.001:
        #     reward = 100
        #     done = True
        #     success = True
        #     return ob, reward, done, success
        # elif delta_dis < 0:
        #     reward = np.exp(-gamma * dis)
        # else:
        #     reward = 0
        #
        # self.rewards.append(reward)
        # # print("######rewards log: ", self.rewards)
        # done = False
        # if len(self.rewards) < 3:
        #     pass
        # elif sum(self.rewards) < 0:
        #     done = True
        # else:
        #     done = False

        """ Reward function 3"""
        # #TODO: consider to loose the target-reached constraint
        # reward = np.exp(-gamma * dis)
        # if dis < 0.05:
        #      # plt.axis('off')
        #      # plt.imshow(ob)
        #      # plt.savefig('/Users/chingandywu/master-thesis/code/src/near/img.png',transparent = True, bbox_inches = 'tight', pad_inches = 0)
        #      # print("##########################Image saved.###########################")
        #      print("Near the target, distance: ", dis)
        # if  dis <= 0.01:
        #     print("#"*50)
        #     print("Target touched!")
        #     print("#"*50)
        #     # reward += 100
        #     self.rewards.clear()
        #     touch = True
        #     done = True
        #     return ob, reward, done, touch
        #
        # if len(self.rewards) < 3:
        #     pass
        # elif np.all(self.rewards > reward):
        #     self.rewards.clear()
        #     done = True
        #     return ob, reward, done, touch
        # else:
        #     done = False
        #
        # self.rewards.append(reward)


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
        # return ob, reward, done, touch
        ######################################
        # return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
<<<<<<< HEAD

=======
>>>>>>> e2c1602f84dfa4a64c0d06b4fc7bb40eb72c7fb3
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
