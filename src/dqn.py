import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense, Input, concatenate
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Sequential, load_model, Model
import os
import parser
import itertools
import csv
# from blob_detector import blob_detector

class DQNAgent:
    #Constructor for the agent (invoked when DQN is first called in main)
    def __init__(self, state_space, action_space):
        self.check_solve = False
        self.render = False  # visualize the training process
        self.action_space = action_space
        #Get size of state and action
        self.state_size = state_space.shape[0]
        self.action_size = len(action_space)
        self.discount_factor = 0.95
        self.learning_rate = 1e-4  # 0.005
        self.epsilon = 0.1 #Fixed
        self.batch_size = 32 #Fixed
        self.memory_size = 500000  # 1000
        self.train_start = 1000 #Fixed
        self.target_update_frequency = 1


        #Number of test states for Q value plots
        self.test_state_no =10000 # 10000

        #Create memory buffer using deque
        self.memory = deque(maxlen=self.memory_size)

        #Create main network and target network (using build_model defined below)
        self.model = self.build_model()
        self.target_model = self.build_model()

        #Initialize target network
        self.update_target_model()

    def build_model(self):

        main_input = Input(shape=(11,))

        # The first branch operates on the main input
        x = Dense(16, activation='relu', kernel_initializer='he_uniform')(main_input)
        x = Dense(16, activation='relu', kernel_initializer='he_uniform')(x)
        x = Dense(self.action_size, activation='linear', kernel_initializer='he_uniform')(x)


        output = Dense(self.action_size, activation='linear', kernel_initializer='he_uniform')(x)
        model = Model(inputs=main_input, output=output)
        opt = Adam(lr=self.learning_rate, decay=1e-6)
        model.compile(loss='mse', optimizer=opt)
        # model.add(Dense(16, input_dim=self.state_size, activation='relu',
        #                 kernel_initializer='he_uniform'))
        # model.add(Dense(16, activation='relu',
        #                 kernel_initializer='he_uniform'))
        # model.add(Dense(self.action_size, activation='linear',
        #                 kernel_initializer='he_uniform'))
        # model.summary()
        # opt = Adam(lr=self.learning_rate, decay=1e-6)
        # model.compile(loss='mse', optimizer=opt)

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
        update_input = np.zeros((batch_size, self.state_size)) #batch_size by state_size two-dimensional array (not matrix!)
        update_target = np.zeros((batch_size, self.state_size)) #Same as above, but used for the target network
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
        action_range = np.linspace(-3.0, 3.0, dis_level)
        self.action_space = list(itertools.product(action_range, action_range))

# class DiscretizedActionWrapper(Env):
#     def __init__(self, env_fns, dis_level):
#         # self.envs = [fn() for fn in env_fns]
#         # env = self.envs[0]
#         Env.__init__(self, len(env_fns), env.observation_space, env.action_space) # adding the key 'episodic_return' because of VecEnv from the module "baselins"
#         self.actions = None
#         if dis_level:
#             action_range = np.linspace(-3.0, 3.0, dis_level)
#             self.action_space = list(itertools.product(action_range, action_range))

def main(args):


    EPISODES = args.episodes
    if args.observation == 'feature':
        env = gym.make('Reacher-v2')
    elif args.observation == 'feature-n-detector':
        env = gym.make('Reacher-v102')
    else:
        print("Observation space not defined")
        quit()
    if args.dis_level is None:
        print("Discretization level not specified")
        quit()
    else:
        env = BasicWrapper(env, args.dis_level)
        print("Action space: ")
        print(env.action_space)
    #Get state and action sizes from the environment
    state_space = env.observation_space
    state_size = env.observation_space.shape[0]
    action_space = env.action_space
    action_size = len(env.action_space)
    # action_size = env.action_space.n
    # action_size = env.action_space.shape[0]


    #Create agent, see the DQNAgent __init__ method for details
    agent = DQNAgent(state_space, action_space)

    #load the pre-trained model
    # path_to_model = 'model_mlp_dtcr.h5'
    # path_to_target = 'target_model_mlp_dtcr.h5'
    # print("#" * 50)
    # if os.path.isfile(path_to_model) and os.path.isfile(path_to_target):
    #     print("Loading the pre-trained model......")
    #     agent.restore_model(path_to_model, path_to_target)
    #     print("Done")
    # else:
    #     print("Pre-trained model doesn't exist.")
    # print("#" * 50)


    #Collect test states for plotting Q values using uniform random policy
    test_states = np.zeros((agent.test_state_no, state_size))
    target_pos_test = np.zeros((agent.test_state_no, 2))
    max_q = np.zeros((EPISODES, agent.test_state_no))
    max_q_mean = np.zeros((EPISODES,1))
    done = True
    for i in range(agent.test_state_no):
        if done:
            done = False
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            test_states[i] = state
        else:
            action_idx = random.randrange(action_size)
            action = env.action_space[action_idx]

            next_state, reward, done, touch= env.step(action)
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
            score += reward #Store episodic reward
            state = next_state #Propagate state

            if done:
                #At the end of very episode, update the target network
                if e % agent.target_update_frequency == 0:
                    agent.update_target_model()
                #Plot the play time for every episode
                scores.append(score)
                episodes.append(e)

                print("episode:", e, "  score:", score," q_value:", max_q_mean[e],"  memory length:",
                      len(agent.memory))

                # if the mean of scores of last 100 episodes is bigger than 195
                # stop training
                if agent.check_solve:
                    last_hundred_q_mean = np.mean(max_q_mean[-min(100, len(max_q_mean)):])
                    if abs(last_hundred_q_mean - max_q_mean[e]) / last_hundred_q_mean  <= 0.01:
                        print("solved after", e-100, "episodes")
                        agent.save_model(path_to_model, path_to_target)
                        print("Models are saved.")
                        agent.plot_data(episodes, scores,max_q_mean[:e+1], success_cnt)
                        sys.exit()


    # Save max q mean to csv file
    save_dir = 'data/dqn_discrete/' + args.observation + '_l'+ str(args.dis_level) + '_q-mean.csv'
    with open(save_dir, mode='a') as log_file:
        writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # print("episodic returns: ", agent.episodic_returns)
        writer.writerow(max_q_mean.flatten())

    # Save max q mean to csv file
    save_dir = 'data/dqn_discrete/' + args.observation + '_l'+ str(args.dis_level) + '_scores.csv'
    with open(save_dir, mode='a') as log_file:
        writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # print("episodic returns: ", agent.episodic_returns)
        writer.writerow(scores)


    # plot data right after training
    # agent.plot_data(episodes, scores, max_q_mean, success_cnt)
    # Save the model
    # agent.save_model(path_to_model, path_to_target)
    env.close()

if __name__ == '__main__':
    """
    Specify the following:
    r: random seed
    e: episodes
    o: observation space
    l: discretization level

    """

    # fix the random seed
    random.seed(parser.args.random_seed)
    np.random.seed(parser.args.random_seed)

    if parser.args.test:
        #TODO: to be filled in
        pass
        # reward = test(parser.args)
        # print("Reward: ", reward)
    else:
        main(parser.args)
