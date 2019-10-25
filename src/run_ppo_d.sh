# franka feature
# discretization level = 5
# python3 ppo_discrete.py -o franka-feature -g 6 -r 1 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-feature -g 6 -r 2 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-feature -g 6 -r 4 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-feature -g 6 -r 8 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-feature -g 6 -r 10 -s 1e5 -len 2048  -l 5
# discretization level = 7
# python3 ppo_discrete.py -o franka-feature -g 6 -r 1 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-feature -g 6 -r 2 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-feature -g 6 -r 4 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-feature -g 6 -r 8 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-feature -g 6 -r 10 -s 1e5 -len 2048  -l 4
# discretization level = 11
# python3 ppo_discrete.py -o franka-feature -g 6 -r 1 -s 1e5 -len 2048  -l 6
# python3 ppo_discrete.py -o franka-feature -g 6 -r 2 -s 1e5 -len 2048  -l 6
# python3 ppo_discrete.py -o franka-feature -g 6 -r 4 -s 1e5 -len 2048  -l 6
# python3 ppo_discrete.py -o franka-feature -g 6 -r 8 -s 1e5 -len 2048  -l 6
# python3 ppo_discrete.py -o franka-feature -g 6 -r 10 -s 1e5 -len 2048  -l 6

# # franka detector
# # discretization level = 5
# python3 ppo_discrete.py -o franka-detector -g 6 -r 1 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-detector -g 6 -r 2 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-detector -g 6 -r 4 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-detector -g 6 -r 8 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-detector -g 6 -r 10 -s 1e5 -len 2048  -l 5
# # discretization level = 7
# python3 ppo_discrete.py -o franka-detector -g 6 -r 1 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-detector -g 6 -r 2 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-detector -g 6 -r 4 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-detector -g 6 -r 8 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-detector -g 6 -r 10 -s 1e5 -len 2048  -l 4
# # discretization level = 11
# python3 ppo_discrete.py -o franka-detector -g 6 -r 1 -s 1e5 -len 2048  -l 6
# python3 ppo_discrete.py -o franka-detector -g 6 -r 2 -s 1e5 -len 2048  -l 6
# python3 ppo_discrete.py -o franka-detector -g 6 -r 4 -s 1e5 -len 2048  -l 6
# python3 ppo_discrete.py -o franka-detector -g 6 -r 8 -s 1e5 -len 2048  -l 6
# python3 ppo_discrete.py -o franka-detector -g 6 -r 10 -s 1e5 -len 2048  -l 6



# franka feature
# python3 ppo_continuous.py -o franka-detector -g 7 -r 1 -s 5e5 -len 2048 -t
# python3 ppo_continuous.py -o franka-detector -g 7 -r 2 -s 5e5 -len 2048 -t
# python3 ppo_continuous.py -o franka-detector -g 7 -r 4 -s 5e5 -len 2048 -t
# python3 ppo_continuous.py -o franka-detector -g 7 -r 8 -s 5e5 -len 2048 -t
# python3 ppo_continuous.py -o franka-detector -g 7 -r 10 -s 5e5 -len 2048 -t



# feature
# discretization level = 5
python3 ppo_discrete.py -o feature -g 4 -r 1 -s 1e5 -len 128  -l 5
python3 ppo_discrete.py -o feature -g 4 -r 2 -s 1e5 -len 128  -l 5
python3 ppo_discrete.py -o feature -g 4 -r 4 -s 1e5 -len 128  -l 5
python3 ppo_discrete.py -o feature -g 4 -r 8 -s 1e5 -len 128  -l 5
python3 ppo_discrete.py -o feature -g 4 -r 10 -s 1e5 -len 128  -l 5
# discretization level = 7
python3 ppo_discrete.py -o feature -g 4 -r 1 -s 1e5 -len 128  -l 7
python3 ppo_discrete.py -o feature -g 4 -r 2 -s 1e5 -len 128  -l 7
python3 ppo_discrete.py -o feature -g 4 -r 4 -s 1e5 -len 128  -l 7
python3 ppo_discrete.py -o feature -g 4 -r 8 -s 1e5 -len 128  -l 7
python3 ppo_discrete.py -o feature -g 4 -r 10 -s 1e5 -len 128  -l 7
# discretization level = 11
python3 ppo_discrete.py -o feature -g 4 -r 1 -s 1e5 -len 128  -l 11
python3 ppo_discrete.py -o feature -g 4 -r 2 -s 1e5 -len 128  -l 11
python3 ppo_discrete.py -o feature -g 4 -r 4 -s 1e5 -len 128  -l 11
python3 ppo_discrete.py -o feature -g 4 -r 8 -s 1e5 -len 128  -l 11
python3 ppo_discrete.py -o feature -g 4 -r 10 -s 1e5 -len 128  -l 11

# # feature-n-detector
# # discretization level = 5
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 1 -s 1e5 -len 128  -l 5
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 2 -s 1e5 -len 128  -l 5
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 4 -s 1e5 -len 128  -l 5
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 8 -s 1e5 -len 128  -l 5
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 10 -s 1e5 -len 128  -l 5
# # discretization level = 7
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 1 -s 1e5 -len 128  -l 7
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 2 -s 1e5 -len 128  -l 7
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 4 -s 1e5 -len 128  -l 7
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 8 -s 1e5 -len 128  -l 7
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 10 -s 1e5 -len 128  -l 7
# # discretization level = 11
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 1 -s 1e5 -len 128  -l 11
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 2 -s 1e5 -len 128  -l 11
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 4 -s 1e5 -len 128  -l 11
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 8 -s 1e5 -len 128  -l 11
# python3 ppo_discrete.py -o feature-n-detector -g 4 -r 10 -s 1e5 -len 128  -l 11
#
#
# # pixel
# # discretization level = 5
# python3 ppo_discrete.py -o pixel -g 6 -r 1 -s 1e5 -len 128  -l 5
# python3 ppo_discrete.py -o pixel -g 6 -r 2 -s 1e5 -len 128  -l 5
# python3 ppo_discrete.py -o pixel -g 6 -r 4 -s 1e5 -len 128  -l 5
# python3 ppo_discrete.py -o pixel -g 6 -r 8 -s 1e5 -len 128  -l 5
# python3 ppo_discrete.py -o pixel -g 6 -r 10 -s 1e5 -len 128  -l 5
# # discretization level = 7
# python3 ppo_discrete.py -o pixel -g 6 -r 1 -s 1e5 -len 128  -l 7
# python3 ppo_discrete.py -o pixel -g 6 -r 2 -s 1e5 -len 128  -l 7
# python3 ppo_discrete.py -o pixel -g 6 -r 4 -s 1e5 -len 128  -l 7
# python3 ppo_discrete.py -o pixel -g 6 -r 8 -s 1e5 -len 128  -l 7
# python3 ppo_discrete.py -o pixel -g 6 -r 10 -s 1e5 -len 128  -l 7
# # discretization level = 11
# python3 ppo_discrete.py -o pixel -g 6 -r 1 -s 1e5 -len 128  -l 11
# python3 ppo_discrete.py -o pixel -g 6 -r 2 -s 1e5 -len 128  -l 11
# python3 ppo_discrete.py -o pixel -g 6 -r 4 -s 1e5 -len 128  -l 11
# python3 ppo_discrete.py -o pixel -g 6 -r 8 -s 1e5 -len 128  -l 11
# python3 ppo_discrete.py -o pixel -g 6 -r 10 -s 1e5 -len 128  -l 11



# franka feature
# discretization level = 5
# python3 ppo_discrete.py -o franka-feature -g 5 -r 1 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-feature -g 5 -r 2 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-feature -g 5 -r 4 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-feature -g 5 -r 8 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-feature -g 5 -r 10 -s 1e5 -len 2048  -l 5
# discretization level = 7
# python3 ppo_discrete.py -o franka-feature -g 5 -r 1 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-feature -g 5 -r 2 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-feature -g 5 -r 4 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-feature -g 5 -r 8 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-feature -g 5 -r 10 -s 1e5 -len 2048  -l 4
# discretization level = 11
# python3 ppo_discrete.py -o franka-feature -g 5 -r 1 -s 1e5 -len 2048  -l 6
# python3 ppo_discrete.py -o franka-feature -g 5 -r 2 -s 1e5 -len 2048  -l 6
# python3 ppo_discrete.py -o franka-feature -g 5 -r 4 -s 1e5 -len 2048  -l 6
# python3 ppo_discrete.py -o franka-feature -g 5 -r 8 -s 1e5 -len 2048  -l 6
# python3 ppo_discrete.py -o franka-feature -g 5 -r 10 -s 1e5 -len 2048  -l 6

# # franka detector
# # discretization level = 5
# python3 ppo_discrete.py -o franka-detector -g 6 -r 1 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-detector -g 6 -r 2 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-detector -g 6 -r 4 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-detector -g 6 -r 8 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-detector -g 6 -r 10 -s 1e5 -len 2048  -l 5
# # discretization level = 7
# python3 ppo_discrete.py -o franka-detector -g 6 -r 1 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-detector -g 6 -r 2 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-detector -g 6 -r 4 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-detector -g 6 -r 8 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-detector -g 6 -r 10 -s 1e5 -len 2048  -l 4
# # discretization level = 11
# python3 ppo_discrete.py -o franka-detector -g 6 -r 1 -s 1e5 -len 2048  -l 6
# python3 ppo_discrete.py -o franka-detector -g 6 -r 2 -s 1e5 -len 2048  -l 6
# python3 ppo_discrete.py -o franka-detector -g 6 -r 4 -s 1e5 -len 2048  -l 6
# python3 ppo_discrete.py -o franka-detector -g 6 -r 8 -s 1e5 -len 2048  -l 6
# python3 ppo_discrete.py -o franka-detector -g 6 -r 10 -s 1e5 -len 2048  -l 6


# franka pixel
# discretization level = 5
# python3 ppo_discrete.py -o franka-pixel -g 6 -r 1 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-pixel -g 6 -r 2 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-pixel -g 6 -r 4 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-pixel -g 6 -r 8 -s 1e5 -len 2048  -l 5
# python3 ppo_discrete.py -o franka-pixel -g 6 -r 10 -s 1e5 -len 2048  -l 5
# discretization level = 7
# python3 ppo_discrete.py -o franka-pixel -g 6 -r 1 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-pixel -g 6 -r 2 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-pixel -g 6 -r 4 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-pixel -g 6 -r 8 -s 1e5 -len 2048  -l 4
# python3 ppo_discrete.py -o franka-pixel -g 6 -r 10 -s 1e5 -len 2048  -l 4
# discretization level = 11
# python3 ppo_discrete.py -o franka-pixel -g 6 -r 1 -s 1e5 -len 2048  -l 3
# python3 ppo_discrete.py -o franka-pixel -g 6 -r 2 -s 1e5 -len 2048  -l 3
# python3 ppo_discrete.py -o franka-pixel -g 6 -r 4 -s 1e5 -len 2048  -l 3
# python3 ppo_discrete.py -o franka-pixel -g 6 -r 8 -s 1e5 -len 2048  -l 3
# python3 ppo_discrete.py -o franka-pixel -g 6 -r 10 -s 1e5 -len 2048  -l 3
