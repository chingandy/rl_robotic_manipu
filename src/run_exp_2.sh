# continuous action space, no discretization level

# franka-feature

# python3 ddpg_continuous.py -o franka-feature -g 3 -r 1 -s 1e6 -v
# python3 ddpg_continuous.py -o franka-feature -g 3 -r 2 -s 1e6
# python3 ddpg_continuous.py -o franka-feature -g 3 -r 4 -s 1e6
# python3 ddpg_continuous.py -o franka-feature -g 3 -r 8 -s 1e6
# python3 ddpg_continuous.py -o franka-feature -g 3 -r 10 -s 1e6


# feature
# python3 ddpg_continuous.py -o feature -g 3 -r 1 -s 1e5 -v
# python3 ddpg_continuous.py -o feature -g 3 -r 2 -s 1e5
# python3 ddpg_continuous.py -o feature -g 3 -r 4 -s 1e5
# python3 ddpg_continuous.py -o feature -g 3 -r 8 -s 1e5
# python3 ddpg_continuous.py -o feature -g 3 -r 10 -s 1e5


# feature-n-detector
# python3 ddpg_continuous.py -o feature-n-detector -g 4 -r 1 -s 1e5 -v
# python3 ddpg_continuous.py -o feature-n-detector -g 4 -r 2 -s 1e5
# python3 ddpg_continuous.py -o feature-n-detector -g 4 -r 4 -s 1e5
# python3 ddpg_continuous.py -o feature-n-detector -g 4 -r 8 -s 1e5
# python3 ddpg_continuous.py -o feature-n-detector -g 4 -r 10 -s 1e5

# pixel
python3 ddpg_continuous.py -o pixel -g 3 -r 1 -s 1e5 -v
python3 ddpg_continuous.py -o pixel -g 3 -r 2 -s 1e5
python3 ddpg_continuous.py -o pixel -g 3 -r 4 -s 1e5
python3 ddpg_continuous.py -o pixel -g 3 -r 8 -s 1e5
python3 ddpg_continuous.py -o pixel -g 3 -r 10 -s 1e5




#=================================================================
# dqn
# discretization level = 5
# python3 dqn.py -e 1000 -o feature -r 1 -l 5
# python3 dqn.py -e 1000 -o feature -r 2 -l 5
# python3 dqn.py -e 1000 -o feature -r 4 -l 5
# python3 dqn.py -e 1000 -o feature -r 8 -l 5
# python3 dqn.py -e 1000 -o feature -r 10 -l 5

# discretization level = 5
# python3 dqn.py -e 1000 -o feature -r 1 -l 7
# python3 dqn.py -e 1000 -o feature -r 2 -l 7
# python3 dqn.py -e 1000 -o feature -r 4 -l 7
# python3 dqn.py -e 1000 -o feature -r 8 -l 7
# python3 dqn.py -e 1000 -o feature -r 10 -l 7

# discretization level = 5
# python3 dqn.py -e 1000 -o feature -r 1 -l 11
# python3 dqn.py -e 1000 -o feature -r 2 -l 11
# python3 dqn.py -e 1000 -o feature -r 4 -l 11
# python3 dqn.py -e 1000 -o feature -r 8 -l 11
# python3 dqn.py -e 1000 -o feature -r 10 -l 11
