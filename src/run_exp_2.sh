# continuous action space, no discretization level
python3 ddpg_continuous.py -o franka-feature -g 3 -r 1 -s 1e6 -v
python3 ddpg_continuous.py -o franka-feature -g 3 -r 2 -s 1e6
python3 ddpg_continuous.py -o franka-feature -g 3 -r 4 -s 1e6
python3 ddpg_continuous.py -o franka-feature -g 3 -r 8 -s 1e6
python3 ddpg_continuous.py -o franka-feature -g 3 -r 10 -s 1e6
