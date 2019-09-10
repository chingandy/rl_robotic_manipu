import argparse
import os
parser = argparse.ArgumentParser()
# parser.add_argument("square", type=int,
#                     help="display a square of a given number")
parser.add_argument("-g", "--gpu", help="specify the ids of gpu is going to be used")

parser.add_argument("-e", "--episodes", type=int,  help="specify the number of episodes")

parser.add_argument("-s", "--num_steps", type=float,  help="specify the maximum number of steps")

parser.add_argument("-l", "--dis_level", type=int,  help="specify the discretization level")

parser.add_argument("-r", "--random_seed", type=int,  help="specify the random seed")

parser.add_argument("-t", "--test", action="store_true", help="test the result and render videos")

parser.add_argument("-v", "--video_rendering", action="store_true", help="render videos") # for ppo

parser.add_argument("-p", "--path", help="specify the directory of the model.")
#parser.add_argument("-s", "--steps", type=int, help="specify the number of steps during the test.")
parser.add_argument("-o", "--observation", help="specify the observation space: pixel, feature_n_detector")

args = parser.parse_args()
# answer = args.square**2
if args.gpu:
    print("Using GPU: {}".format(args.gpu))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
