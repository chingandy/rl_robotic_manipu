# rl_robotic_mani
## How to install the enviroment
1. First of all, you need a MuJoCo license to use MuJoCo.
2. Follow the instruction here:<https://github.com/openai/mujoco-py#install-mujoco>(Install MuJoCo).
3. Install MuJoCo:
  `pip3 install -U 'mujoco-py==2.0.2.0a1'​`
4. Trouble shooting: you may have to add some path to your environment variables.

   ex:
   ```
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-396
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chingan/.mujoco/mujoco200/bin`
   ```
   If encounter Python locale error:
   ```
   export LC_ALL="en_US.UTF-8"
   export LC_CTYPE="en_US.UTF-8"
   sudo dpkg-reconfigure locales
   ```
5. Install gym:
   * Change the mujoco_py version in setup.py to 2.0.2.0a1
   * Install gym from source:
     `pip3 install -e '.[mujoco]'`
## How to export the videos recording the training process?
First, import the module `wrappers`
```
from gym import wrappers
from time import time
```
Then, after create the environment, include the line as the following:
```
env = gym.make(ENV_NAME)
env = wrappers.Monitor(env, './videos/' + str(time()) + '/')
```
The mp4 files will save to `./videos`.

We can also include the argument `force=True` to replace your existing recording with the current recording.
```
env = wrappers.Monitor(env, './videos/' + str(time()) + '/')
```
problem: this seems to affect the setting of the customized environment.
```
Traceback (most recent call last):
  File "cnn_w_dtcr.py", line 343, in <module>
    main(parser.args)
  File "cnn_w_dtcr.py", line 282, in main
    state = env.reset() #Initialize/reset the environment
  File "/home/chingan/thesis/rl_robotic_manipu/deform_manipu/gym/gym/wrappers/monitor.py", line 37, in reset
    self._before_reset()
  File "/home/chingan/thesis/rl_robotic_manipu/deform_manipu/gym/gym/wrappers/monitor.py", line 180, in _before_reset
    self.stats_recorder.before_reset()
  File "/home/chingan/thesis/rl_robotic_manipu/deform_manipu/gym/gym/wrappers/monitoring/stats_recorder.py", line 68, in before_reset
    raise error.Error("Tried to reset environment which is not done. While the monitor is active for {}, you cannot call reset() unless the episode is over.".format(self.env_id))
gym.error.Error: Tried to reset environment which is not done. While the monitor is active for Reacher-v101, you cannot call reset() unless the episode is over.
```
## How to change the max steps per episode?

You can find the variable `max_episode_steps` in the directory `gym/gym/envs/__init__.py`.


# Reacher v2 (OpenAI Gym)
## Overview

### Details
* Name: Reacher-v2
* Category: Mujoco
* [Leaderboard Page]()
* Old links:
  * [Environment Page](https://gym.openai.com/envs/Reacher-v2/)


### Description
A 2 DOF robotic arm whose task is to reacher a particular target in the field.
### Source
To be completed

## Environment

### Observation

Type: Box(11,)

Num | Observation | Min | Max
---|---|---|---
0 | cos(theta) (joint 0)  | None | None
1 | cos(theta) (joint 1) | cos(-3.0)  | cos(3.0)
2 | sin(theta) (joint 0)  | None | None
3 | sin(theta) (joint 1) | sin(-3.0) | sin(3.0)
4 | qpos (the x coordinate of the target ) |  |
5 | qpos (the y coordinate of the target ) |  |
6 | qvel (the velocity of the fingertip in the x direction ) |  |
7 | qvel (the velocity of the fingertip in the y direction ) |  |
8 | the x-axis component of the vector from the target to the fingertip |  |
9 | the y-axis component of the vector from the target to the fingertip |  |
10 | the z-axis component of the vector from the target to the fingertip |  |
### Actions
<!---
Type: Discrete(2)

Num | Action
--- | ---
0 | Push cart to the left
1 | Push cart to the right
-->


### Reward


### Starting State

### Episode Termination


### Solved Requirements
