import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
import keras
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, concatenate
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Sequential, load_model, Model
import tensorflow as tf
import  keras.backend.tensorflow_backend as K
import os
# import parser
import argparse
import itertools
import csv
from gym import wrappers
from time import time
import  keras.backend.tensorflow_backend as K
from keras.utils.training_utils import multi_gpu_model
# config = tf.ConfigProto(log_device_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# K.set_session(sess)

os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 6} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

inshape = (256, 256, 3)



class DQNAgent:
    #Constructor for the agent (invoked when DQN is first called in main)
    def __init__(self, state_space, action_space):
        self.check_solve = False
        self.render = False  # visualize the training process
        self.action_space = action_space
        #Get size of state and action
        self.state_size = state_space.shape[0]
        self.action_size = len(action_space)
        self.discount_factor = 0.99
        self.learning_rate = 1e-4  # 0.005
        self.epsilon = 0.1 #Fixed
        self.batch_size = 32 #Fixed
        self.memory_size = 500000  # 1000
        self.train_start = 32 #Fixed
        self.target_update_frequency = 1 # 1


        #Number of test states for Q value plots
        self.test_state_no =10000 # 10000

        #Create memory buffer using deque
        self.memory = deque(maxlen=self.memory_size)

        #Create main network and target network (using build_model defined below)
        self.model = self.build_model()
        self.model = multi_gpu_model(self.model, 2)
        self.target_model = self.build_model()
        self.target_model = multi_gpu_model(self.target_model, 2)

        #Initialize target network
        self.update_target_model()

    def build_model(self):


        main_input = Input(shape=inshape)
        # aux_input = Input(shape=(2,))

        # The first branch operates on the main input
        x = Conv2D(64, kernel_size=5, strides=(2,2), activation='relu')(main_input)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Conv2D(32, kernel_size=5, strides=(2,2), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Conv2D(16, kernel_size=5, strides=(2,2), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Flatten()(x)

        output = Dense(4096, activation='relu')(x)
        output = Dense(256)(output)
        output = Dense(self.action_size)(output)
        model = Model(inputs=main_input, output=output)
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    #After some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    #Get action from model using epsilon-greedy policy
    def get_action(self, state):

        if np.random.rand() <= self.epsilon:
            action =  random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            action =  np.argmax(q_value[0])
        return action

    def get_test_action(self, state):
        q_value = self.model.predict(state)
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
        update_input = np.zeros((batch_size, inshape[0], inshape[1], inshape[2])) #batch_size by state_size two-dimensional array (not matrix!)
        update_target = np.zeros((batch_size, inshape[0], inshape[1], inshape[2])) #Same as above, but used for the target network
        action, reward, done = [], [], [] #Empty arrays that will grow dynamically
        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]#Allocate s(i) to the network input array from iteration i in the batch
            action.append(mini_batch[i][1]) #Store a(i)
            reward.append(mini_batch[i][2]) #Store r(i)
            update_target[i] = mini_batch[i][3] #Allocate s'(i) for the target network array from iteration i in the batch
            done.append(mini_batch[i][4])  #Store done(i)

        target = self.model.predict(update_input) #Generate target values for training the inner loop network using the network model
        target_val = self.target_model.predict(update_target) #Generate the target values for training the outer loop target network
        #Q Learning: get maximum Q value at s' from target network
        for i in range(self.batch_size):
            action_ind = self.action_space.index(action[i])
            if done[i]:
                target[i][action_ind]= reward[i]
            else:
                target[i][action_ind] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        #Train the inner loop network
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
        return


    def save_model(self, path_to_model, path_to_target):
            self.model.save(path_to_model)
            self.target_model.save(path_to_target)
            return

    def restore_model(self, path_to_model, path_to_target):
            self.model = load_model(path_to_model)
            self.target_model = load_model(path_to_target)
            return

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
            action_range = np.linspace(-3.0, 3.0, dis_level)
            self.action_space = list(itertools.product(action_range, action_range))

def main(args):


    EPISODES = args.episodes
    if args.observation == 'pixel':
        env = gym.make('Reacher-v101')
    else:
        print("Observation not specified")
        quit()


    # discretization level of action space
    if args.dis_level is None:
        print("Discretization level not specified")
        quit()
    else:
        env = BasicWrapper(env, args.dis_level)
        print("Action space: ")
        print(env.action_space)

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

    # the directory to save the models
    path_to_model = './models/model_pixel.h5'
    path_to_target = './models/target_model_pixel.h5'


    #Collect test states for plotting Q values using uniform random policy
    test_states = np.zeros((agent.test_state_no, inshape[0], inshape[1], inshape[2]))
    max_q = np.zeros((EPISODES, agent.test_state_no))
    max_q_mean = np.zeros((EPISODES,1))
    done = True
    for i in range(agent.test_state_no):
        if done:
            done = False
            state = env.reset()
            # state = np.reshape(state, [1, state_size])
            test_states[i] = state
        else:
            action_idx = random.randrange(action_size)
            action = env.action_space[action_idx]

            next_state, reward, done, touch= env.step(action)
            # next_state = np.reshape(next_state, [1, state_size])
            test_states[i] = state
            state = next_state

    # scores, episodes, success_cnt = [], [], [] #Create dynamically growing score and episode counters
    scores, episodes = [], [] #Create dynamically growing score and episode counters

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset() #Initialize/reset the environment
        state = np.expand_dims(state, axis=0)
        #Compute Q values for plotting
        tmp = agent.model.predict(test_states)  # tmp.shape = num of test states * num of actions
        max_q[e][:] = np.max(tmp, axis=1)
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
        save_dir = 'data/dqn_test/' + args.observation + '_l'+ str(args.dis_level) + '_q-mean.csv'
        with open(save_dir, mode='a') as log_file:
            writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # print("episodic returns: ", agent.episodic_returns)
            writer.writerow(max_q_mean.flatten())

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


    # plot data right after training
    # agent.plot_data(episodes, scores, max_q_mean, success_cnt)
    # Save the model
    agent.save_model(path_to_model, path_to_target)
    env.close()

if __name__ == '__main__':
    """
    Specify the following:
    r: random seed
    e: episodes
    o: observation space
    l: discretization level
    t: test mode (optional, logging to the test folder)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--episodes", type=int,  help="specify the number of episodes")
    parser.add_argument("-r", "--random_seed", type=int,  help="specify the random seed")
    parser.add_argument("-o", "--observation", help="specify the observation space: pixel, feature_n_detector")
    parser.add_argument("-l", "--dis_level", type=int,  help="specify the discretization level")
    parser.add_argument("-t", "--test", action="store_true", help="test the result and render videos")
    parser.add_argument("-v", "--video_rendering", action="store_true", help="render videos") # for ppo

    args = parser.parse_args()




    # fix the random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    main(args)
    # # fix the random seed
    # random.seed(parser.args.random_seed)
    # np.random.seed(parser.args.random_seed)
    #
    # main(parser.args)
