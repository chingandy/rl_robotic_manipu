# franka detector
# discretization level = 5
python3 ppo_discrete.py -o franka-detector -g 7 -r 1 -s 1e5 -len 2048  -l 5
python3 ppo_discrete.py -o franka-detector -g 7 -r 2 -s 1e5 -len 2048  -l 5
python3 ppo_discrete.py -o franka-detector -g 7 -r 4 -s 1e5 -len 2048  -l 5
python3 ppo_discrete.py -o franka-detector -g 7 -r 8 -s 1e5 -len 2048  -l 5
python3 ppo_discrete.py -o franka-detector -g 7 -r 10 -s 1e5 -len 2048  -l 5
# discretization level = 7
python3 ppo_discrete.py -o franka-detector -g 7 -r 1 -s 1e5 -len 2048  -l 4
python3 ppo_discrete.py -o franka-detector -g 7 -r 2 -s 1e5 -len 2048  -l 4
python3 ppo_discrete.py -o franka-detector -g 7 -r 4 -s 1e5 -len 2048  -l 4
python3 ppo_discrete.py -o franka-detector -g 7 -r 8 -s 1e5 -len 2048  -l 4
python3 ppo_discrete.py -o franka-detector -g 7 -r 10 -s 1e5 -len 2048  -l 4
# discretization level = 11
python3 ppo_discrete.py -o franka-detector -g 7 -r 1 -s 1e5 -len 2048  -l 6
python3 ppo_discrete.py -o franka-detector -g 7 -r 2 -s 1e5 -len 2048  -l 6
python3 ppo_discrete.py -o franka-detector -g 7 -r 4 -s 1e5 -len 2048  -l 6
python3 ppo_discrete.py -o franka-detector -g 7 -r 8 -s 1e5 -len 2048  -l 6
python3 ppo_discrete.py -o franka-detector -g 7 -r 10 -s 1e5 -len 2048  -l 6






# ddpg continuous

# python3 ddpg_continuous.py -o franka-pixel -g 4 -r 1 -s 6e5 -t
# python3 ddpg_continuous.py -o franka-pixel -g 4 -r 2 -s 6e5 -t
# python3 ddpg_continuous.py -o franka-pixel -g 4 -r 4 -s 6e5 -t
# python3 ddpg_continuous.py -o franka-pixel -g 4 -r 8 -s 6e5 -t
# python3 ddpg_continuous.py -o franka-pixel -g 4 -r 10 -s 6e5 -t






# ppo continuous

# franka feature
# python3 ppo_continuous.py -o franka-feature -g 6 -r 1 -s 5e5 -len 2048
# python3 ppo_continuous.py -o franka-feature -g 4 -r 2 -s 5e5 -len 2048
# python3 ppo_continuous.py -o franka-feature -g 4 -r 4 -s 5e5 -len 2048
# python3 ppo_continuous.py -o franka-feature -g 4 -r 8 -s 5e5 -len 2048
# python3 ppo_continuous.py -o franka-feature -g 4 -r 10 -s 5e5 -len 2048

# franka detector
# python3 ppo_continuous.py -o franka-detector -g 4 -r 1 -s 5e5 -len 2048
# python3 ppo_continuous.py -o franka-detector -g 4 -r 2 -s 5e5 -len 2048
# python3 ppo_continuous.py -o franka-detector -g 4 -r 4 -s 5e5 -len 2048
# python3 ppo_continuous.py -o franka-detector -g 4 -r 8 -s 5e5 -len 2048
# python3 ppo_continuous.py -o franka-detector -g 4 -r 10 -s 5e5 -len 2048

# franka pixel
# python3 ppo_continuous.py -o franka-pixel -g 5 -r 1 -s 5e5 -len 2048
# python3 ppo_continuous.py -o franka-pixel -g 5 -r 2 -s 5e5 -len 2048
# python3 ppo_continuous.py -o franka-pixel -g 5 -r 4 -s 5e5 -len 2048
# python3 ppo_continuous.py -o franka-pixel -g 5 -r 8 -s 5e5 -len 2048
# python3 ppo_continuous.py -o franka-pixel -g 5 -r 10 -s 5e5 -len 2048
