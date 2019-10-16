# franka feature
python3 ppo_continuous.py -o franka-feature -g 7 -r 1 -s 5e5 -len 2048 -v
python3 ppo_continuous.py -o franka-feature -g 7 -r 2 -s 5e5 -len 2048
python3 ppo_continuous.py -o franka-feature -g 7 -r 4 -s 5e5 -len 2048
python3 ppo_continuous.py -o franka-feature -g 7 -r 8 -s 5e5 -len 2048
python3 ppo_continuous.py -o franka-feature -g 7 -r 10 -s 5e5 -len 2048

# franka feature
python3 ppo_continuous.py -o franka-detector -g 7 -r 1 -s 5e5 -len 2048
python3 ppo_continuous.py -o franka-detector -g 7 -r 2 -s 5e5 -len 2048
python3 ppo_continuous.py -o franka-detector -g 7 -r 4 -s 5e5 -len 2048
python3 ppo_continuous.py -o franka-detector -g 7 -r 8 -s 5e5 -len 2048
python3 ppo_continuous.py -o franka-detector -g 7 -r 10 -s 5e5 -len 2048


# franka pixel
# python3 ppo_continuous.py -o franka-pixel -g 7 -r 1 -s 5e5 -len 2048 -t
# python3 ppo_continuous.py -o franka-pixel -g 6 -r 2 -s 5e5 -len 2048 -t
# python3 ppo_continuous.py -o franka-pixel -g 5 -r 4 -s 5e5 -len 2048 -t
# python3 ppo_continuous.py -o franka-pixel -g 4 -r 8 -s 5e5 -len 2048 -t
# python3 ppo_continuous.py -o franka-pixel -g 3 -r 10 -s 5e5 -len 2048 -t
