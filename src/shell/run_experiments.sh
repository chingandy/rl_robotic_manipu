#

python3 ppo_discrete.py -o feature-n-detector -g 5 -r 1 -s 1e5 -len 128  -l 5
python3 ppo_discrete.py -o feature-n-detector -g 5 -r 2 -s 1e5 -len 128  -l 5
python3 ppo_discrete.py -o feature-n-detector -g 5 -r 4 -s 1e5 -len 128  -l 5
python3 ppo_discrete.py -o feature-n-detector -g 5 -r 8 -s 1e5 -len 128  -l 5
python3 ppo_discrete.py -o feature-n-detector -g 5 -r 10 -s 1e5 -len 128 -l 5

python3 ppo_discrete.py -o feature-n-detector -g 5 -r 1 -s 1e5 -len 128  -l 7
python3 ppo_discrete.py -o feature-n-detector -g 5 -r 2 -s 1e5 -len 128  -l 7
python3 ppo_discrete.py -o feature-n-detector -g 5 -r 4 -s 1e5 -len 128  -l 7
python3 ppo_discrete.py -o feature-n-detector -g 5 -r 8 -s 1e5 -len 128  -l 7
python3 ppo_discrete.py -o feature-n-detector -g 5 -r 10 -s 1e5 -len 128 -l 7

python3 ppo_discrete.py -o feature-n-detector -g 5 -r 1 -s 1e5 -len 128  -l 11
python3 ppo_discrete.py -o feature-n-detector -g 5 -r 2 -s 1e5 -len 128  -l 11
python3 ppo_discrete.py -o feature-n-detector -g 5 -r 4 -s 1e5 -len 128  -l 11
python3 ppo_discrete.py -o feature-n-detector -g 5 -r 8 -s 1e5 -len 128  -l 11
python3 ppo_discrete.py -o feature-n-detector -g 5 -r 10 -s 1e5 -len 128 -l 11




#
python3 ppo_discrete.py -o feature -g 5 -r 1 -s 1e5 -len 128  -l 5
python3 ppo_discrete.py -o feature -g 5 -r 2 -s 1e5 -len 128  -l 5
python3 ppo_discrete.py -o feature -g 5 -r 4 -s 1e5 -len 128  -l 5
python3 ppo_discrete.py -o feature -g 5 -r 8 -s 1e5 -len 128  -l 5
python3 ppo_discrete.py -o feature -g 5 -r 10 -s 1e5 -len 128 -l 5

python3 ppo_discrete.py -o feature -g 5 -r 1 -s 1e5 -len 128  -l 7
python3 ppo_discrete.py -o feature -g 5 -r 2 -s 1e5 -len 128  -l 7
python3 ppo_discrete.py -o feature -g 5 -r 4 -s 1e5 -len 128  -l 7
python3 ppo_discrete.py -o feature -g 5 -r 8 -s 1e5 -len 128  -l 7
python3 ppo_discrete.py -o feature -g 5 -r 10 -s 1e5 -len 128 -l 7

python3 ppo_discrete.py -o feature -g 5 -r 1 -s 1e5 -len 128  -l 11
python3 ppo_discrete.py -o feature -g 5 -r 2 -s 1e5 -len 128  -l 11
python3 ppo_discrete.py -o feature -g 5 -r 4 -s 1e5 -len 128  -l 11
python3 ppo_discrete.py -o feature -g 5 -r 8 -s 1e5 -len 128  -l 11
python3 ppo_discrete.py -o feature -g 5 -r 10 -s 1e5 -len 128 -l 11




# ppo_continuous feature

# continuous action space, no discretization level len = 128
# python3 ppo_continuous.py -o feature -g 4 -r 1 -s 5e5 -v
# python3 ppo_continuous.py -o feature -g 4 -r 2 -s 5e5
# python3 ppo_continuous.py -o feature -g 4 -r 4 -s 5e5
# python3 ppo_continuous.py -o feature -g 4 -r 8 -s 5e5
# python3 ppo_continuous.py -o feature -g 4 -r 10 -s 5e5

# continuous action space rollout = 1024
# python3 ppo_continuous.py -o feature -g 4 -r 1 -s 5e5 -len 1024
# python3 ppo_continuous.py -o feature -g 4 -r 2 -s 5e5 -len 1024
# python3 ppo_continuous.py -o feature -g 4 -r 4 -s 5e5 -len 1024
# python3 ppo_continuous.py -o feature -g 4 -r 8 -s 5e5  -len 1024
# python3 ppo_continuous.py -o feature -g 4 -r 10 -s 5e5 -len 1024


"""
old reward function

"""

# python3 ddpg_continuous.py -o franka-pixel -g 4 -r 1 -s 1e5 -v
# python3 ddpg_continuous.py -o franka-pixel -g 4 -r 2 -s 1e5
# python3 ddpg_continuous.py -o franka-pixel -g 4 -r 4 -s 1e5
# python3 ddpg_continuous.py -o franka-pixel -g 4 -r 8 -s 1e5
# python3 ddpg_continuous.py -o franka-pixel -g 4 -r 10 -s 1e5


# python3 ddpg_continuous.py -o franka-detector -g 4 -r 1 -s 1e5 -v
# python3 ddpg_continuous.py -o franka-detector -g 4 -r 2 -s 1e5
# python3 ddpg_continuous.py -o franka-detector -g 4 -r 4 -s 1e5
# python3 ddpg_continuous.py -o franka-detector -g 4 -r 8 -s 1e5
# python3 ddpg_continuous.py -o franka-detector -g 4 -r 10 -s 1e5
#======================================================================
# ppo_continuous feature-n-detector
# python3 ppo_continuous.py -o feature-n-detector -g 4 -r 1 -s 5e5 -v
# python3 ppo_continuous.py -o feature-n-detector -g 4 -r 2 -s 5e5
# python3 ppo_continuous.py -o feature-n-detector -g 4 -r 4 -s 5e5
# python3 ppo_continuous.py -o feature-n-detector -g 4 -r 8 -s 5e5
# python3 ppo_continuous.py -o feature-n-detector -g 4 -r 10 -s 5e5


#==================================================================
# ppo_continuous feature

# continuous action space, no discretization level
# python3 ppo_continuous.py -o feature -g 4 -r 1 -s 5e5 -v
# python3 ppo_continuous.py -o feature -g 4 -r 2 -s 5e5
# python3 ppo_continuous.py -o feature -g 4 -r 4 -s 5e5
# python3 ppo_continuous.py -o feature -g 4 -r 8 -s 5e5
# python3 ppo_continuous.py -o feature -g 4 -r 10 -s 5e5


#===================================================================
# ppo_discrete  pixel
# discretization level = 5
# python3 ppo_discrete.py -o pixel -g 3 -r 1 -s 1e5  -l 5
# python3 ppo_discrete.py -o pixel -g 3 -r 2 -s 1e5  -l 5
# python3 ppo_discrete.py -o pixel -g 3 -r 4 -s 1e5  -l 5
# python3 ppo_discrete.py -o pixel -g 3 -r 8 -s 1e5  -l 5
# python3 ppo_discrete.py -o pixel -g 3 -r 10 -s 1e5  -l 5

# discretization level = 7
# python3 ppo_discrete.py -o pixel -g 3 -r 1 -s 1e5  -l 7
# python3 ppo_discrete.py -o pixel -g 3 -r 2 -s 1e5  -l 7
# python3 ppo_discrete.py -o pixel -g 3 -r 4 -s 1e5  -l 7
# python3 ppo_discrete.py -o pixel -g 3 -r 8 -s 1e5  -l 7
# python3 ppo_discrete.py -o pixel -g 3 -r 10 -s 1e5  -l 7

# discretization level = 11
# python3 ppo_discrete.py -o pixel -g 3 -r 1 -s 1e5  -l 11
# python3 ppo_discrete.py -o pixel -g 3 -r 2 -s 1e5  -l 11
# python3 ppo_discrete.py -o pixel -g 3 -r 4 -s 1e5  -l 11
# python3 ppo_discrete.py -o pixel -g 3 -r 8 -s 1e5  -l 11
# python3 ppo_discrete.py -o pixel -g 3 -r 10 -s 1e5  -l 11


#======================================================================================
# ppo_discrete  feature
# discretization level = 5
# python3 ppo_discrete.py -o feature -g 3 -r 1 -s 1e5  -l 5
# python3 ppo_discrete.py -o feature -g 3 -r 2 -s 1e5  -l 5
# python3 ppo_discrete.py -o feature -g 3 -r 4 -s 1e5  -l 5
# python3 ppo_discrete.py -o feature -g 3 -r 8 -s 1e5  -l 5
# python3 ppo_discrete.py -o feature -g 3 -r 10 -s 1e5  -l 5

# discretization level = 7
# python3 ppo_discrete.py -o feature -g 3 -r 1 -s 1e5  -l 7
# python3 ppo_discrete.py -o feature -g 3 -r 2 -s 1e5  -l 7
# python3 ppo_discrete.py -o feature -g 3 -r 4 -s 1e5  -l 7
# python3 ppo_discrete.py -o feature -g 3 -r 8 -s 1e5  -l 7
# python3 ppo_discrete.py -o feature -g 3 -r 10 -s 1e5  -l 7

# discretization level = 11
# python3 ppo_discrete.py -o feature -g 3 -r 1 -s 1e5  -l 11
# python3 ppo_discrete.py -o feature -g 3 -r 2 -s 1e5  -l 11
# python3 ppo_discrete.py -o feature -g 3 -r 4 -s 1e5  -l 11
# python3 ppo_discrete.py -o feature -g 3 -r 8 -s 1e5  -l 11
# python3 ppo_discrete.py -o feature -g 3 -r 10 -s 1e5  -l 11






#=======================================================================================
# ppo_discrete feature-n-detector

# discretization level = 5
# python3 ppo_discrete.py -o feature-n-detector -g 3 -r 1 -s 1e5  -l 5
# python3 ppo_discrete.py -o feature-n-detector -g 3 -r 2 -s 1e5  -l 5
# python3 ppo_discrete.py -o feature-n-detector -g 3 -r 4 -s 1e5  -l 5
# python3 ppo_discrete.py -o feature-n-detector -g 3 -r 8 -s 1e5  -l 5
# python3 ppo_discrete.py -o feature-n-detector -g 3 -r 10 -s 1e5  -l 5

# discretization level = 7
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 1 -s 1e5  -l 7
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 2 -s 1e5  -l 7
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 4 -s 1e5  -l 7
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 8 -s 1e5  -l 7
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 10 -s 1e5  -l 7

# discretization level = 11
# python3 ppo_discrete.py -o feature-n-detector -g 5 -r 1 -s 1e5  -l 11
# python3 ppo_discrete.py -o feature-n-detector -g 5 -r 2 -s 1e5  -l 11
# python3 ppo_discrete.py -o feature-n-detector -g 5 -r 4 -s 1e5  -l 11
# python3 ppo_discrete.py -o feature-n-detector -g 5 -r 8 -s 1e5  -l 11
# python3 ppo_discrete.py -o feature-n-detector -g 5 -r 10 -s 1e5  -l 11
