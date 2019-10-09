# continuous action space, no discretization level

# feature
# python3 ddpg_continuous.py -o feature -g 7 -r 1 -s 1e5   -v
# python3 ddpg_continuous.py -o feature -g 7 -r 2 -s 1e5
# python3 ddpg_continuous.py -o feature -g 7 -r 4 -s 1e5
# python3 ddpg_continuous.py -o feature -g 7 -r 8 -s 1e5
# python3 ddpg_continuous.py -o feature -g 7 -r 10 -s 1e5


# feature-n-detector
python3 ddpg_continuous.py -o feature-n-detector -g 7 -r 1 -s 1e5
python3 ddpg_continuous.py -o feature-n-detector -g 7 -r 2 -s 1e5
python3 ddpg_continuous.py -o feature-n-detector -g 7 -r 4 -s 1e5
python3 ddpg_continuous.py -o feature-n-detector -g 7 -r 8 -s 1e5
python3 ddpg_continuous.py -o feature-n-detector -g 7 -r 10 -s 1e5

# pixel
# python3 ddpg_continuous.py -o pixel -g 7 -r 1 -s 1e5
# python3 ddpg_continuous.py -o pixel -g 7 -r 2 -s 1e5
# python3 ddpg_continuous.py -o pixel -g 7 -r 4 -s 1e5
# python3 ddpg_continuous.py -o pixel -g 7 -r 8 -s 1e5
# python3 ddpg_continuous.py -o pixel -g 7 -r 10 -s 1e5


# franka-feature

# python3 ddpg_continuous.py -o franka-feature -g 7 -r 1 -s 1e5
# python3 ddpg_continuous.py -o franka-feature -g 7 -r 2 -s 1e5
# python3 ddpg_continuous.py -o franka-feature -g 7 -r 4 -s 1e5
# python3 ddpg_continuous.py -o franka-feature -g 7 -r 8 -s 1e5
# python3 ddpg_continuous.py -o franka-feature -g 7 -r 10 -s 1e5

# franka-feature

# python3 ddpg_continuous.py -o franka-detector -g 7 -r 1 -s 1e5
# python3 ddpg_continuous.py -o franka-detector -g 7 -r 2 -s 1e5
# python3 ddpg_continuous.py -o franka-detector -g 7 -r 4 -s 1e5
# python3 ddpg_continuous.py -o franka-detector -g 7 -r 8 -s 1e5
# python3 ddpg_continuous.py -o franka-detector -g 7 -r 10 -s 1e5

# franka-pixel
# python3 ddpg_continuous.py -o franka-pixel -g 7 -r 1 -s 1e5
# python3 ddpg_continuous.py -o franka-pixel -g 7 -r 2 -s 1e5
# python3 ddpg_continuous.py -o franka-pixel -g 7 -r 4 -s 1e5
# python3 ddpg_continuous.py -o franka-pixel -g 7 -r 8 -s 1e5
# python3 ddpg_continuous.py -o franka-pixel -g 7 -r 10 -s 1e5




#=================================================================
# dqn
# discretization level = 5
# python3 dqn.py -e 1000 -o feature -r 1 -l 5 -v
# python3 dqn.py -e 1000 -o feature -r 2 -l 5
# python3 dqn.py -e 1000 -o feature -r 4 -l 5
# python3 dqn.py -e 1000 -o feature -r 8 -l 5
# python3 dqn.py -e 1000 -o feature -r 10 -l 5
#
# # discretization level = 5
# python3 dqn.py -e 1000 -o feature -r 1 -l 7 -v
# python3 dqn.py -e 1000 -o feature -r 2 -l 7
# python3 dqn.py -e 1000 -o feature -r 4 -l 7
# python3 dqn.py -e 1000 -o feature -r 8 -l 7
# python3 dqn.py -e 1000 -o feature -r 10 -l 7
#
# # discretization level = 5
# python3 dqn.py -e 1000 -o feature -r 1 -l 11 -v
# python3 dqn.py -e 1000 -o feature -r 2 -l 11
# python3 dqn.py -e 1000 -o feature -r 4 -l 11
# python3 dqn.py -e 1000 -o feature -r 8 -l 11
# python3 dqn.py -e 1000 -o feature -r 10 -l 11



# dqn
# discretization level = 5
# python3 dqn.py -e 1000 -o feature-n-detector -r 1 -l 5
# python3 dqn.py -e 1000 -o feature-n-detector -r 2 -l 5
# python3 dqn.py -e 1000 -o feature-n-detector -r 4 -l 5
# python3 dqn.py -e 1000 -o feature-n-detector -r 8 -l 5
# python3 dqn.py -e 1000 -o feature-n-detector -r 10 -l 5

# discretization level = 5
# python3 dqn.py -e 1000 -o feature-n-detector -r 1 -l 7
# python3 dqn.py -e 1000 -o feature-n-detector -r 2 -l 7
# python3 dqn.py -e 1000 -o feature-n-detector -r 4 -l 7
# python3 dqn.py -e 1000 -o feature-n-detector -r 8 -l 7
# python3 dqn.py -e 1000 -o feature-n-detector -r 10 -l 7

# discretization level = 5
# python3 dqn.py -e 1000 -o feature-n-detector -r 1 -l 11
# python3 dqn.py -e 1000 -o feature-n-detector -r 2 -l 11
# python3 dqn.py -e 1000 -o feature-n-detector -r 4 -l 11
# python3 dqn.py -e 1000 -o feature-n-detector -r 8 -l 11
# python3 dqn.py -e 1000 -o feature-n-detector -r 10 -l 11



# dqn feature
# discretization level = 5
python3 dqn_test.py -s 1e5 -o feature -r 1 -l 5 -g 6 -v
python3 dqn_test.py -s 1e5 -o feature -r 2 -l 5 -g 6
python3 dqn_test.py -s 1e5 -o feature -r 4 -l 5 -g 6
python3 dqn_test.py -s 1e5 -o feature -r 8 -l 5 -g 6
python3 dqn_test.py -s 1e5 -o feature -r 10 -l 5 -g 6

# discretization level = 7
python3 dqn_test.py -s 1e5 -o feature -r 1 -l 7 -g 6
python3 dqn_test.py -s 1e5 -o feature -r 2 -l 7 -g 6
python3 dqn_test.py -s 1e5 -o feature -r 4 -l 7 -g 6
python3 dqn_test.py -s 1e5 -o feature -r 8 -l 7 -g 6
python3 dqn_test.py -s 1e5 -o feature -r 10 -l 7 -g 6

# discretization level = 11
python3 dqn_test.py -s 1e5 -o feature -r 1 -l 11 -g 6
python3 dqn_test.py -s 1e5 -o feature -r 2 -l 11 -g 6
python3 dqn_test.py -s 1e5 -o feature -r 4 -l 11 -g 6
python3 dqn_test.py -s 1e5 -o feature -r 8 -l 11 -g 6
python3 dqn_test.py -s 1e5 -o feature -r 10 -l 11 -g 6

# dqn feature_n_detector
# dqn pixel
# discretization level = 5
python3 dqn_test.py -s 1e5 -o feature-n-detector -r 1 -l 5 -g 6
python3 dqn_test.py -s 1e5 -o feature-n-detector -r 2 -l 5 -g 6
python3 dqn_test.py -s 1e5 -o feature-n-detector -r 4 -l 5 -g 6
python3 dqn_test.py -s 1e5 -o feaeture-n-detector -r 8 -l 5 -g 6
python3 dqn_test.py -s 1e5 -o feature-n-detector -r 10 -l 5 -g 6

# discretization level = 7
python3 dqn_test.py -s 1e5 -o feature-n-detector -r 1 -l 7 -g 6
python3 dqn_test.py -s 1e5 -o feature-n-detector -r 2 -l 7 -g 6
python3 dqn_test.py -s 1e5 -o feature-n-detector -r 4 -l 7 -g 6
python3 dqn_test.py -s 1e5 -o feature-n-detector -r 8 -l 7 -g 6
python3 dqn_test.py -s 1e5 -o feature-n-detector -r 10 -l 7 -g 6

# discretization level = 11
python3 dqn_test.py -s 1e5 -o feature-n-detector -r 1 -l 11 -g 6
python3 dqn_test.py -s 1e5 -o feature-n-detector -r 2 -l 11 -g 6
python3 dqn_test.py -s 1e5 -o feature-n-detector -r 4 -l 11 -g 6
python3 dqn_test.py -s 1e5 -o feature-n-detector -r 8 -l 11 -g 6
python3 dqn_test.py -s 1e5 -o feature-n-detector -r 10 -l 11 -g 6
