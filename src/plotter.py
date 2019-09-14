import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import argparse



def load_data(file_dir):
    data = []
    with open(file_dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row = [float(x) for x in row]
            data.append(row)
    return data


def plot_data(file_list, data_type):

# file_dir = './test.csv'
    colors = ['r', 'g', 'b', 'k', 'c', 'y', 'm'] # matplotlib built-in colors
    linestyles = ['-', '--', ':', '-.', '*']
    fig = plt.figure()
    for i in range(len(file_list)):
        data = load_data(file_list[i])
        # use seaborn to plot the data with five different random seeds
        sns.tsplot(data=data, color=colors[i], linestyle=linestyles[i], ci="sd")
        # our y-axis is "success rate" here
    if data_type == 'episodic':
        plt.ylabel("Episodic returns", fontsize=10)
        plt.xlabel("Episodes", fontsize=10, labelpad=-1)
    elif data_type == 'avg':
        plt.ylabel("Average returns", fontsize=10)
        plt.xlabel("Time steps", fontsize=10, labelpad=-1)

    # our x-axis is iteration number

    # our task is called "Awesome Robot Performance"
    # plt.title("Awesome Robot Performance", fontsize=30)
    # Legend
    plt.legend(labels=['bins = 5', 'bins = 7', 'bins = 11'], loc='lower right', markerscale=11)
    # Show the plot on the screen
    fig.tight_layout()
    plt.savefig("./pic/result.png")
    plt.show()



if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="specify the model type")
    args = parser.parse_args()
    data_type = 'episodic' # 'avg' or 'episodic'

    if args.model == 'ppo_continuous':
        # ppo continuous
        folder_dir = 'data/ppo_continuous/'
        file_list = ['franka-feature.csv']
        file_list = [folder_dir + x for x in file_list]
        plot_data(file_list, data_type)

    elif args.model == 'ppo_discrete':
        # ppo discrete
        folder_dir = 'data/ppo_discrete/'
        # file_list = ['feature_l5.csv','feature_l7.csv','feature_l11.csv']
        file_list = ['pixel_l5.csv','pixel_l7.csv','pixel_l11.csv']
        file_list = [folder_dir + x for x in file_list]
        plot_data(file_list, data_type)


    elif args.model == 'dqn_discrete':
        # dqn
        folder_dir = 'data/dqn_discrete/'
        file_list = ['feature_l5_q-mean.csv', 'feature_l7_q-mean.csv', 'feature_l11_q-mean.csv']
        file_list = [folder_dir + x for x in file_list]
        plot_data(file_list, data_type)

    elif args.model == 'ddpg_continuous':
        # ddpg
        folder_dir = 'data/ddpg_continuous/'
        file_list = ['franka-feature.csv']
        file_list = [folder_dir + x for x in file_list]
        plot_data(file_list, data_type)
    else:
        print("Model not specified.")
