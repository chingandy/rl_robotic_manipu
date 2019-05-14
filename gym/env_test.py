import gym
env = gym.make('FetchPush-v1')
# env.reset()
# for _ in range(10000000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()


""" check the action and observation spaces"""
print(env.action_space)
print(env.observation_space) # Box(3, ) and Box(11, )


# print(env.observation_space.high)
# print(env.observation_space.low)
