import sys
import gym
from gym import wrappers
import pylab
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import parser
import itertools
import csv
from utils import *
from time import time
import pdb




DEVICE = torch.device('cuda:0,1')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNet(nn.Module):

    def __init__(self, dim_input, action_size):
        super().__init__()
        self.fc_1 = nn.Linear(dim_input, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_3 = nn.Linear(32, 16)
        self.logits = nn.Linear(16, action_size)
        # self.to(device=DEVICE)


    def forward(self, x):
        y = F.relu(self.fc_1(x))
        y = F.relu(self.fc_2(y))
        y = F.relu(self.fc_3(y))
        y = self.logits(y)
        return y

# def layer_init(layer, w_scale=1.0):
#     nn.init.orthogonal_(layer.weight.data)
#     layer.weight.data.mul_(w_scale)
#     nn.init.constant_(layer.bias.data, 0)
#     return layer

class QConvNet(nn.Module): # for franka-pixel

    def __init__(self, action_size, in_channels=3):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, 8, kernel_size=7, stride=3)
        self.conv_2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.conv_3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.fc_4 = nn.Linear(39 * 18 * 16, 11232) # 44928
        self.logits = nn.Linear(11232, action_size)
        # self.conv_1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=3)
        # self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        # self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.fc_4 = nn.Linear(39 * 18 * 64, 44928) # 44928
        # self.logits = nn.Linear(44928, action_size)

        # self.to(device=DEVICE)

    def forward(self, x):
        # print("X: ", type(x))
        x = x.permute(0, 3, 1, 2)
        y = F.relu(self.conv_1(x))
        y = F.relu(self.conv_2(y))
        y = F.relu(self.conv_3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc_4(y))
        y = self.logits(y)
        return y

# model = QNet(23, 10)
# print(model)
# params = list(model.parameters())
# print(len(params))
# print(params[0].size())
# print(params[0].type())
#
# model_2 = QNet(23, 10)
# model_2.load_state_dict(model.state_dict())
# print(model_2)
# # print("Model state dict: ")
# # print(model.state_dict())
#
# quit()





class DQNAgent:
    #Constructor for the agent (invoked when DQN is first called in main)
    def __init__(self, state_space, action_space):
        self.check_solve = False
        self.render = False  # visualize the training process
        self.action_space = action_space
        #Get size of state and action
        self.state_size = state_space.shape[0]
        self.action_size = len(action_space)
        self.discount_factor = 0.99 # 0.95
        self.learning_rate = 1e-4 if parser.args.learning_rate is None else parser.args.learning_rate  # 0.005
        self.epsilon = 0.1 #Fixed
        self.batch_size = 32 #Fixed
        self.memory_size = 500000  # 1000
        self.train_start = 32 #Fixed
        self.target_update_frequency = 1 # 1

        #Number of test states for Q value plots
        self.test_state_no =100 # 10000

        #Create memory buffer using deque
        self.memory = deque(maxlen=self.memory_size)

        #Create main network and target network (using build_model defined below)
        self.model = self.build_model()
        self.target_model = self.build_model()

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate) # optimizer for self.model
        #Initialize target network
        self.update_target_model()

    def build_model(self):

        if parser.args.observation == 'franka-pixel':
            model = QConvNet(self.action_size)
        else:
            model = QNet(self.state_size, self.action_size)

        # print("Model params: ")
        # print(list(model.parameters()))
        # print("Size: ", sys.getsizeof(list(model.parameters())))
        # quit()
        # if torch.cuda.device_count() > 1:
        #     print("#" * 100)
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #     model = nn.DataParallel(model)

        model.to(DEVICE)

        return model

    #After some time interval update the target model to be same with model
    def update_target_model(self):
        # self.target_model.set_weights(self.model.get_weights())
        self.target_model.load_state_dict(self.model.state_dict())

    #Get action from model using epsilon-greedy policy
    def get_action(self, state):

        if np.random.rand() <= self.epsilon:
            action =  random.randrange(self.action_size)
        else:
            state = torch.tensor(state, device=DEVICE, dtype=torch.float32)
            q_value = self.model(state)
            action =  np.argmax(to_np(q_value[0]))
        return action

    def get_test_action(self, state):
        q_value = self.model(state)
        action =  np.argmax(q_value[0])
        return action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #Add sample to the end of the list

    #Sample <s,a,r,s'> from replay memory
    def train_model(self):
        if len(self.memory) < self.train_start: #Do not train if not enough memory
            return
        batch_size = min(self.batch_size, len(self.memory)) #Train on at most as many samples as you have in memory
        mini_batch = random.sample(self.memory, batch_size) #Uniformly sample the memory buffer


        #Preallocate network and target network input matrices.
        if parser.args.observation == 'franka-pixel':
            update_input = np.zeros((batch_size, inshape[0], inshape[1], inshape[2]))
            update_target = np.zeros((batch_size, inshape[0], inshape[1], inshape[2]))
        else:
            update_input = np.zeros((batch_size, self.state_size)) #batch_size by state_size two-dimensional array (not matrix!)
            update_target = np.zeros((batch_size, self.state_size)) #Same as above, but used for the target network
        action, reward, done = [], [], [] #Empty arrays that will grow dynamically
        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]#Allocate s(i) to the network input array from iteration i in the batch
            action.append(mini_batch[i][1]) #Store a(i)
            reward.append(mini_batch[i][2]) #Store r(i)
            update_target[i] = mini_batch[i][3] #Allocate s'(i) for the target network array from iteration i in the batch
            done.append(mini_batch[i][4])  #Store done(i)
        del mini_batch

        update_input = torch.tensor(update_input, device=DEVICE, dtype=torch.float32)
        target = self.model(update_input) #Generate target values for training the inner loop network using the network model
        update_target = torch.tensor(update_target, device=DEVICE, dtype=torch.float32)
        target_val = self.target_model(update_target) #Generate the target values for training the outer loop target network
        #Q Learning: get maximum Q value at s' from target network
        for i in range(self.batch_size):
            action_ind = self.action_space.index(action[i])
            if done[i]:
                target[i][action_ind]= reward[i]
            else:
                target[i][action_ind] = reward[i] + self.discount_factor * (
                    np.amax(to_np(target_val[i])))

        #Train the inner loop network
        # self.model.fit(update_input, target, batch_size=self.batch_size,
        #                epochs=1, verbose=0)
        q = self.model(update_input)
        loss = (target - q).pow(2).mul(0.5).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return


    # def save_model(self, path_to_model, path_to_target):
    #         self.model.save(path_to_model)
    #         self.target_model.save(path_to_target)
    #         return

    # def restore_model(self, path_to_model, path_to_target):
    #         self.model = load_model(path_to_model)
    #         self.target_model = load_model(path_to_target)
    #         return

    #Plots the score per episode as well as the maximum q value per episode, averaged over precollected states.
    def plot_data(self, episodes, scores, max_q_mean):
        pylab.figure(0)
        pylab.plot(episodes, max_q_mean, 'b')
        pylab.xlabel("Episodes")
        pylab.ylabel("Average Q Value")
        pylab.savefig("pic/qvalues.png")

        # pylab.figure(1)
        # pylab.plot(episodes, scores, 'b')
        # pylab.xlabel("Episodes")
        # pylab.ylabel("Score")
        # pylab.savefig("pic/scores.png")
        #
        #
        # pylab.figure(2)
        # pylab.plot(episodes, success_cnt, 'b')
        # pylab.xlabel("Episodes")
        # pylab.ylabel("Successes")
        # pylab.savefig("pic/successes.png")


class BasicWrapper(gym.Wrapper):
    def __init__(self, env, dis_level):
        super().__init__(env)
        self.env = env
        if dis_level == -1:
            self.action_space = [[0.1, 0], [-0.1, 0], [0, -0.1], [0, 0.1]]
        else:
            action_range = np.linspace(-2.0, 2.0, dis_level)
            # print("self.action-space: ", self.action_space.shape[0])
            self.action_space = list(itertools.product(action_range, repeat=self.action_space.shape[0]))

def main(args):


    EPISODES = args.episodes

    # observation
    if args.observation == 'feature':
        env = gym.make('Reacher-v2')
    elif args.observation == 'feature-n-detector':
        env = gym.make('Reacher-v102')

    elif args.observation == 'franka-feature':
        env = gym.make("FrankaReacher-v0")

    elif args.observation == 'franka-detector':
        env = gym.make("FrankaReacher-v1")

    elif args.observation == 'franka-pixel':
        env = gym.make("FrankaReacher-v2")

    else:
        print("Observation space not defined")
        quit()

    # discretization level of action space
    if args.dis_level is None:
        print("Discretization level not specified")
        quit()
    else:
        env = BasicWrapper(env, args.dis_level)
        # print("Action space: ")
        # print(env.action_space)

    # video recording
    if args.video_rendering:
        env = wrappers.Monitor(env, './videos/' + str(time()) + '/')

    #Get state and action sizes from the environment
    state_space = env.observation_space
    state_size = env.observation_space.shape[0]
    action_space = env.action_space
    action_size = len(env.action_space)

    #Create agent, see the DQNAgent __init__ method for details
    agent = DQNAgent(state_space, action_space)




    #Collect test states for plotting Q values using uniform random policy
    if args.observation == 'franka-pixel':
        test_states = np.zeros((agent.test_state_no, inshape[0], inshape[1], inshape[2]))
    else:
        test_states = np.zeros((agent.test_state_no, state_size))
    # target_pos_test = np.zeros((agent.test_state_no, 2))
    max_q = np.zeros((EPISODES, agent.test_state_no))
    max_q_mean = np.zeros((EPISODES,1))
    done = True
    for i in range(agent.test_state_no):
        if done:
            done = False
            state = env.reset()
            if args.observation == 'franka-pixel':
                pass
            else:
                state = np.reshape(state, [1, state_size])
            test_states[i] = state
        else:
            action_idx = random.randrange(action_size)
            action = env.action_space[action_idx]

            next_state, reward, done, touch= env.step(action)
            if args.observation == 'franka-pixel':
                pass
            else:
                next_state = np.reshape(next_state, [1, state_size])
            test_states[i] = state
            state = next_state

    # scores, episodes, success_cnt = [], [], [] #Create dynamically growing score and episode counters
    scores, episodes = [], [] #Create dynamically growing score and episode counters

    for e in range(EPISODES):
        score = 0
        state = env.reset() #Initialize/reset the environment
        done = False
        state = np.expand_dims(state, axis=0)

        #Compute Q values for plotting
        test_states = torch.tensor(test_states, device=DEVICE, dtype=torch.float32)
        # print("test_states: ", type(test_states))
        # quit()
        tmp = agent.model(test_states)  # tmp.shape = num of test states * num of actions
        # pdb.set_trace()
        # test = torch.max(tmp, dim=1)

        max_q[e][:] = to_np(torch.max(tmp, dim=1)[0].detach())

        max_q_mean[e] = np.mean(max_q[e][:])


        while not done:

            if agent.render:
                env.render() #Show cartpole animation

            #Get action for the current state and go one step in environment
            ###################################
            action_idx = agent.get_action(state)
            action = env.action_space[action_idx]
            ###################################
            next_state, reward, done, info= env.step(action)
            next_state = np.expand_dims(next_state, axis=0) #Reshape next_state similarly to state

            #Save sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            #Training step
            agent.train_model()
            score = reward +  0.99 * score#Store episodic reward
            state = next_state #Propagate state

            if done:
                #At the end of very episode, update the target network
                if e % agent.target_update_frequency == 0:
                    agent.update_target_model()
                #Plot the play time for every episode
                scores.append(score)
                episodes.append(e)

                print("episode:", e, "  score:", score," q_value:", max_q_mean[e],"  memory length:", len(agent.memory))


    if args.test:

        # Save max q mean to csv file
        # save_dir = 'data/dqn_test/' + args.observation + '_l'+ str(args.dis_level) + '_q-mean.csv'
        # with open(save_dir, mode='a') as log_file:
        #     writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     # print("episodic returns: ", agent.episodic_returns)
        #     writer.writerow(max_q_mean.flatten())

        #Save scores to csv file
        save_dir = 'data/dqn_test/' + args.observation + '_l'+ str(args.dis_level) + '_scores.csv'
        with open(save_dir, mode='a') as log_file:
            writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # print("episodic returns: ", agent.episodic_returns)
            writer.writerow(scores)
    else:
    # Save max q mean to csv file
        save_dir = 'data/dqn_discrete/' + args.observation + '_l'+ str(args.dis_level) + '_q-mean.csv'
        with open(save_dir, mode='a') as log_file:
            writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # print("episodic returns: ", agent.episodic_returns)
            writer.writerow(max_q_mean.flatten())

        #Save scores to csv file
        save_dir = 'data/dqn_discrete/' + args.observation + '_l'+ str(args.dis_level) + '_scores.csv'
        with open(save_dir, mode='a') as log_file:
            writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # print("episodic returns: ", agent.episodic_returns)
            writer.writerow(scores)



if __name__ == '__main__':
    """
    Specify the following:
    r: random seed
    e: episodes
    o: observation space
    l: discretization level
    t: test mode (optional, logging to the test folder)
    """

    # fix the random seed
    random.seed(parser.args.random_seed)
    np.random.seed(parser.args.random_seed)

    if parser.args.observation == 'franka-pixel':
        inshape = (256, 128, 3)
    elif parser.args.observation == 'pixel':
        inshape = (128, 128, 3)

    main(parser.args)
