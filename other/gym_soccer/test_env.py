import gym
env = gym.make("gym_soccer:soccer-v0")
state = env.reset()
print("good good ")
print(env.observation_space)
