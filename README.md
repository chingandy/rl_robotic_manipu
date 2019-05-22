# rl_robotic_manipu
## Note: how to install the enviroment
1. First of all, you need a MuJoCo license to use MuJoCo.
2. Follow the instruction here:<https://github.com/openai/mujoco-py#install-mujoco>(Install MuJoCo).
3. Install MuJoCo:
  `pip3 install -U 'mujoco-py==2.0.2.0a1'​`
4. Trouble shooting: you may have to add some path to your environment variables.
   ex:` export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-396,export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chingan/.mujoco/mujoco200/bin`
5. Install gym: 
   * Change the mujoco_py version in setup.py to 2.0.2.0a1
   * Install gym from source:
     `pip3 install -e '.[mujoco]'`

