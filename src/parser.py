import argparse
import os
parser = argparse.ArgumentParser()
# parser.add_argument("square", type=int,
#                     help="display a square of a given number")
parser.add_argument("-g", "--gpu", help="specify the ids of gpu is going to be used")

parser.add_argument("-e", "--episodes", type=int,  help="specify the number of episodes")

parser.add_argument("-t", "--test", action="store_true", help="test the result and render videos")

parser.add_argument("-p", "--path", help="specify the directory of the model.")
#parser.add_argument("-s", "--steps", type=int, help="specify the number of steps during the test.")
args = parser.parse_args()
# answer = args.square**2
if args.gpu:
    print("Using GPU: {}".format(args.gpu))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
