import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
from baselines.common.running_mean_std import RunningMeanStd
from envs import *
import numpy as np
from utils import *
from skimage.io import imsave
from parser import *
import csv
from torchsummary import summary
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



##################################################################
class BaseNet:
    def __init__(self):
        pass
class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        y = self.fc_head(phi)
        return y
##########################################################################
# TODO: look into nn.init.orthogonal_
def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer
##########################################################################
class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x

class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=7, stride=3))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(39 * 39 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x
##########################################################################

class DQNActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.start()

    def _transition(self):
        print("#" * 100)
        print("In _transition ...")
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        with config.lock:
            q_values = self._network(config.state_normalizer(self._state))
        q_values = to_np(q_values).flatten()
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)


        # real_action = action[0]
        # real_action = self.task.action_space[real_action]
        # print("real action: ", real_action)
        # quit()
        # next_state, reward, done, info = self._task.step(real_action)
        # entry = [self._state[0], real_action, reward[0], next_state[0], int(done[0]), info]
        next_state, reward, done, info = self._task.step([action])
        entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        return entry


class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)
        self.episodic_returns = []

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        transitions = self.actor.step()
        # print("Transitions: ", len(transitions))
        # print("Transitions: ", len(transitions))
        # print(transitions)
        # quit()
        experiences = []
        for state, action, reward, next_state, done, info in transitions:

            # if info[0]['episodic_return']:
            #     self.episodic_returns.append(info[0]['episodic_return'])
            print("##########################")
            real_action = action[0]
            real_action = self.task.action_space[real_action]
            self.record_online_return(info)
            self.total_steps += 1
            reward = config.reward_normalizer(reward)
            experiences.append([state, action, reward, next_state, done])
        self.replay.feed_batch(experiences)

        if self.total_steps > self.config.exploration_steps:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            q_next = self.target_network(next_states).detach()
            q_next = q_next.max(1)[0]
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).long()
            q = self.network(states)
            q = q[self.batch_indices, actions]
            loss = (q_next - q).pow(2).mul(0.5).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

def dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.video_rendering = False
    config.num_workers = 1
    config.dis_level = args.dis_level if args.dis_level is not None else 7
    config.task_fn = lambda: Task(config.game, config.video_rendering, config.dis_level, num_envs=config.num_workers)
    config.eval_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    # config.network_fn = lambda: DuelingNet(config.action_dim, FCBody(config.state_dim))
    # config.replay_fn = lambda: Replay(memory_size=int(1e4), batch_size=10)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e4), batch_size=10)

    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 1000
    # config.double_q = True
    config.double_q = False
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.eval_interval = args.rollout_length   # int(5e3)
    config.max_steps = 1e5 if args.num_steps is None else args.num_steps
    config.async_actor = False
    agent = DQNAgent(config)
    run_steps(agent)

    # Save the episodic return to csv file
    # save_dir = 'data/dqn_test/' + args.observation + '_test_l'+ str(args.dis_level) + '.csv'
    # with open(save_dir, mode='a') as log_file:
    #     writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     # print("episodic returns: ", agent.episodic_returns)
    #     writer.writerow(agent.episodic_returns)

    # Save the episodic return to csv file
    save_dir = 'data/dqn_test/' + args.observation + '_l'+ str(args.dis_level) + '.csv'
    with open(save_dir, mode='a') as log_file:
        writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # print("episodic returns: ", agent.episodic_returns)
        writer.writerow(agent.episodic_returns)




    # plot the episodic returns
    # import pylab
    # pylab.figure(0)
    # pylab.plot(agent.episodic_returns, 'b')
    # pylab.xlabel("Episodes")
    # pylab.ylabel("Episodic return")
    # if args.observation == 'feature_n_detector':
    #     pylab.savefig("pic/ppo_discrete/feature_n_detector.png")
    # elif args.observation == 'cart':
    #     pylab.savefig("pic/ppo_discrete/cartpole.png")
    # elif args.observation == 'mountain-car':
    #     pylab.savefig("pic/ppo_discrete/mountain-car.png")






if __name__ == '__main__':

    """
    specify the following:
    o: observation space
    r: random seed
    l: discretization level
    g: gpu
    s: step (optional)
    len: rollout length

    """
    mkdir('log')
    mkdir('tf_log')
    mkdir('data')
    set_one_thread()
    print("Random seed: ", args.random_seed)
    print("Discretization level: ", args.dis_level)
    random_seed(args.random_seed)
    select_device(0) # select_device(gpu_id)


    if args.observation == 'feature':
        env = 'Reacher-v2'
        dqn_feature(game=env)

    elif args.observation == 'feature-n-detector':
        # print("argument parser works")
        env = 'Reacher-v102'
        dqn_feature(game=env)

    elif args.observation == 'pixel':
        env = "Reacher-v101"
        ppo_pixel(game=env)

    elif args.observation == 'franka-feature':
        env = "FrankaReacher-v0"
        dqn_feature(game=env)
    elif args.observation == 'franka-detector':
        env = "FrankaReacher-v1"
        ppo_feature(game=env)
    else:
        print("Observation space isn't specified.")
    # elif args.observation == "cart":
    #     env = "CartPole-v0"
    #     ppo_feature(game=env)
    # elif args.observation == "mountain-car":
    #     env = "MountainCar-v0"
    #     ppo_feature(game=env)
    # else:
    #     print("Observation space isn't specified.")
