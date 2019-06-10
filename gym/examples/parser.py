import argparse
import os
parser = argparse.ArgumentParser()
# parser.add_argument("square", type=int,
#                     help="display a square of a given number")
parser.add_argument("-g", "--gpu",
                    help="specify the ids of gpu is going to be used")
args = parser.parse_args()
# answer = args.square**2
if args.gpu:
    print("Using GPU: {}".format(args.gpu))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
