from gym.envs.registration import register

register(
    id='soccer-v0',
    entry_point='gym_soccer.envs:SoccerEnv',
)
