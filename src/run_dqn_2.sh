# dqn: log evaluation returns instead of online returns

# pixel
# discretization level = 5
python3 dqn_pixel.py -r 1 -e 200 -o pixel -l 5 -t
python3 dqn_pixel.py -r 2 -e 200 -o pixel -l 5 -t
python3 dqn_pixel.py -r 4 -e 200 -o pixel -l 5 -t
python3 dqn_pixel.py -r 8 -e 200 -o pixel -l 5 -t
python3 dqn_pixel.py -r 10 -e 200 -o pixel -l 5 -t

# discretization level = 7
python3 dqn_pixel.py -r 1 -e 200 -o pixel -l 7 -t
python3 dqn_pixel.py -r 2 -e 200 -o pixel -l 7 -t
python3 dqn_pixel.py -r 4 -e 200 -o pixel -l 7 -t
python3 dqn_pixel.py -r 8 -e 200 -o pixel -l 7 -t
python3 dqn_pixel.py -r 10 -e 200 -o pixel -l 7 -t

# discretization level = 5
python3 dqn_pixel.py -r 1 -e 200 -o pixel -l 11 -t
python3 dqn_pixel.py -r 2 -e 200 -o pixel -l 11 -t
python3 dqn_pixel.py -r 4 -e 200 -o pixel -l 11 -t
python3 dqn_pixel.py -r 8 -e 200 -o pixel -l 11 -t
python3 dqn_pixel.py -r 10 -e 200 -o pixel -l 11 -t
