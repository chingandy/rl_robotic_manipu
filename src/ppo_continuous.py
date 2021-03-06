import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from baselines.common.running_mean_std import RunningMeanStd
from envs import *
import numpy as np
from utils import *
from skimage.io import imsave
from parser import *
import csv

class Storage:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r', 'm',
                       'v', 'q', 'pi', 'log_pi', 'ent',
                       'adv', 'ret', 'q_a', 'log_pi_a',
                       'mean']
        self.keys = keys
        self.size = size
        self.reset()

    def add(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)

class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x

# class NatureConvBody(nn.Module):
#     def __init__(self, in_channels=4):
#         super(NatureConvBody, self).__init__()
#         self.feature_dim = 512
#         self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
#         self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
#         self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
#         self.fc4 = layer_init(nn.Linear(28 * 28 * 64, self.feature_dim))
#
#     def forward(self, x):
#         y = F.relu(self.conv1(x))
#         y = F.relu(self.conv2(y))
#         y = F.relu(self.conv3(y))
#         y = y.view(y.size(0), -1)
#         y = F.relu(self.fc4(y))
#         return y

class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        if args.observation == 'pixel':
            self.feature_dim = 512
            self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=7, stride=3))
            self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2))
            self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
            self.fc4 = layer_init(nn.Linear(18 * 18 * 64, self.feature_dim))
        elif args.observation == 'franka-pixel':
            self.feature_dim = 512
            self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=7, stride=3))
            self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2))
            self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
            self.fc4 = layer_init(nn.Linear(39 * 18 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


class BaseNet:
    def __init__(self):
        pass

# TODO: look into nn.init.orthogonal_
def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data) # fills the input Tensor with a orthogonal matrix
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0) # fills the input Tensor with the value in the second argument, 0 in this case
    return layer


class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim) # no use in this case actually
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim) # actor (policy)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim) # critic
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3) # fully-connected layer for the probability distribution over actions
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3) # fully-connected layer for the state-value function, output: one dimension

        """ the parameters for the actor and the crititic """
        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

        self.std = nn.Parameter(torch.zeros(action_dim))
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        if args.observation == 'pixel' or args.observation == 'franka-pixel':
            obs = obs.permute(0, 3, 1, 2)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)

        mean = torch.tanh(self.fc_action(phi_a))
        dist = torch.distributions.Normal(mean, F.softplus(self.std)) # distribution over actions

        v = self.fc_critic(phi_v) # state-value function

        if action is None:
            action = dist.sample()  # e.g. action = tensor([[-0.0010, -0.0970]]) , torch.Size([1,2])
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'mean': mean,
                'v': v}

class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])]) # create the input-output dimension pairs
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0

    def close(self):
        close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state):
        raise NotImplementedError

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        return ret

    def eval_episodes(self):
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
            self.total_steps, np.mean(episodic_returns), np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        ))
        self.logger.add_scalar('episodic_return_test', np.mean(episodic_returns), self.total_steps)
        return {
            'episodic_return_test': np.mean(episodic_returns),
        }

    def record_online_return(self, info, offset=0):
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, ret))
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def switch_task(self):
        config = self.config
        if not config.tasks:
            return
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)

    def record_episode(self, dir, env):
        mkdir(dir)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, dir, steps)
            action = self.record_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            steps += 1
            if ret is not None:
                break

    def record_step(self, state):
        raise NotImplementedError

    # For DMControl
    def record_obs(self, env, dir, steps):
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave('%s/%04d.png' % (dir, steps), obs)

class PPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.episodic_returns = []
        self.returns = []
        self.avg_returns = []
    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)

        """ collect a trajectory D, trajectory length = config.rollout_length """
        states = self.states # initial state
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            if info[0]['episodic_return'] is not None:
                self.episodic_returns.append(info[0]['episodic_return'])
            # save each
            self.record_online_return(info)
            #TODO: check out these normalizers

            rewards = config.reward_normalizer(rewards)  # this does't have any effect on the rewards unless reward_normalizer is specified.
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1), # to decide where the state is a terminal state
                         's': tensor(states)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
        storage.add(prediction)
        storage.placeholder()

        """ Calculate the advantage function for each state """
        advantages = tensor(np.zeros((config.num_workers, 1))) # create a placeholder for advantage estimates
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns # rewards-to-go

            """ log the returns for plots """
            self.returns.append(returns.item())
            self.avg_returns.append(sum(self.returns) / len(self.returns))

            if not config.use_gae:  # TODO: check out general advantage estimate
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach() # store the advantage of each state in one rollout
            storage.ret[i] = returns.detach() # store the discounted sum of rewards

        states, actions, log_probs_old, returns, advantages = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        # advantages = (advantages - advantages.mean()) / advantages.std()  # normalize the advantages => do we really need this? seems like it doesn't help a lot

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size) # create a generator of random indices in batches, states.size: 2048, mini_batch_size: 64
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]   # this method of slicing only works in pytorch tensors
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = self.network(sampled_states, sampled_actions)  # GaussianActorCriticNet
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages # .clamp() is also a special function in pytorch
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()


                # value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()
                value_loss = 1 * (sampled_returns - prediction['v']).pow(2).mean()  # follow the value in the ppo paper


                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip) # do gradient clipping in-place
                self.opt.step()

def ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, config.video_rendering)
    config.eval_env = config.task_fn()
    if args.observation == 'pixel' or args.observation == 'franka-pixel':
        config.network_fn = lambda: GaussianActorCriticNet(
            config.state_dim, config.action_dim, phi_body=NatureConvBody(in_channels=3)
        )
    else:
        config.network_fn = lambda: GaussianActorCriticNet(
            config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, gate=torch.tanh),
            critic_body=FCBody(config.state_dim, gate=torch.tanh))
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 128  if args.rollout_length is None else args.rollout_length  # 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.max_steps = 3000 if args.num_steps is None else args.num_steps
    config.state_normalizer = MeanStdNormalizer()
    config.video_rendering = args.video_rendering# set to False if no need to render videos

    agent = PPOAgent(config)
    run_steps(agent)

    if args.test:
        # Save the episodic return to csv file
        save_dir = 'data/ppo_continuous_test/' + args.observation + '.csv'
        with open(save_dir, mode='a') as log_file:
            writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # print("episodic returns: ", agent.episodic_returns)
            writer.writerow(agent.episodic_returns)
    else:
        # Save the episodic return to csv file
        save_dir = 'data/ppo_continuous/' + args.observation + '.csv'
        with open(save_dir, mode='a') as log_file:
            writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # print("episodic returns: ", agent.episodic_returns)
            writer.writerow(agent.episodic_returns)

        # Save the return for each step
        save_dir = 'data/ppo_continuous/' + args.observation + '_avg-returns.csv'
        with open(save_dir, mode='a') as log_file:
            writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # print("episodic returns: ", agent.episodic_returns)
            writer.writerow(agent.avg_returns)




if __name__ == '__main__':


    """
    specify the following:
    o: observation space
    r: random seed
    g: gpu
    s: step (optional)
    v: render videos (default: False)
    len: rollout length (default: 128)
    t: if you are just testing another setting, this flag make sure the data
       won't be corrupted
    """
    mkdir('log')
    mkdir('tf_log')
    mkdir('data')
    set_one_thread()
    print("Random seed: ", args.random_seed)
    random_seed(args.random_seed)
    select_device(0) # select_device(gpu_id)


    # Reacher environments
    if args.observation == 'feature':
        env = 'Reacher-v2'
        ppo_continuous(game=env)

    elif args.observation == 'feature-n-detector':
        # print("argument parser works")
        env = 'Reacher-v102'
        ppo_continuous(game=env)

    elif args.observation == 'pixel':
        env = "Reacher-v101"
        ppo_continuous(game=env)


    # FrankaReacher environments
    elif args.observation == 'franka-feature':
        env = 'FrankaReacher-v0'
        ppo_continuous(game=env)

    elif args.observation == 'franka-detector':
        env = 'FrankaReacher-v1'
        ppo_continuous(game=env)

    elif args.observation == 'franka-pixel':
        env = 'FrankaReacher-v2'
        ppo_continuous(game=env)

    else:
        print("Observation space isn't specified.")
    # mkdir('log')
    # mkdir('tf_log')
    # set_one_thread()
    # random_seed()
    # select_device(0)
    # env = "Reacher-v2"
    # # env = "FrankaReacher-v0"
    # print("#" * 100)
    # print("Env: ", env)
    # print("#" * 100)
    # ppo_continuous(game=env)
