import gym
env = gym.make("Reacher-v101").unwrapped
print(env.action_space)
print(env.action_dim)
