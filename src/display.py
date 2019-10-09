import numpy as np
import matplotlib.pyplot as plt
import csv
# import os


def calculate_mean(data_list):
    return np.mean(data_list[-50:])

def load_data(file_dir):
    data = []
    means = []
    with open(file_dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row = [float(x) for x in row]
            means.append(calculate_mean(row))
            data.append(row)
    return data, np.mean(means)


if __name__ == '__main__':

    """
    Note: aware the number of episodes taken into account when the mean is calculated.
    """

    # show the mean of 100 elements of each row
    folder_dir = 'data/ppo_continuous/'
    file_list = ['franka-feature_len2048.csv']
    # file_list = ['franka-detector_l5.csv', 'franka-detector_l7.csv', 'franka-detector_l11.csv']
    # file_list = ['feature-n-detector_len128_l5.csv', 'feature-n-detector_len128_l7.csv', 'feature-n-detector_len128_l11.csv']
    file_list = [folder_dir + x for x in file_list]


    means = []
    for i in range(len(file_list)):
        data, mean = load_data(file_list[i])
        means.append(mean)
    print("Means of three different discretization levels: ", means)
    print("Mean over all: ", np.mean(means))
