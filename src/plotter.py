import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import argparse
import os



def load_data(file_dir, head):
    data = []
    with open(file_dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if head is None:
                row = [float(x) for x in row]
            else:
                row = [float(row[i]) for i in range(head)]
            data.append(row)
    return data


def plot_data(file_list, data_type, action_type, head=None):

# file_dir = './test.csv'
    colors = ['r', 'g', 'b', 'k', 'c', 'y', 'm'] # matplotlib built-in colors
    linestyles = ['-', '--', ':', '-.', '*']
    fig = plt.figure()
    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)
    for i in range(len(file_list)):
        data = load_data(file_list[i], head)
        # use seaborn to plot the data with five different random seeds
        sns.tsplot(data=data, color=colors[i], linestyle=linestyles[i], ci="sd")
        # our y-axis is "success rate" here
    if data_type == 'episodic':
        plt.ylabel("Episodic returns", fontsize=15)
        plt.xlabel("Episodes", fontsize=15, labelpad=-1)
    elif data_type == 'avg':
        plt.ylabel("Average returns", fontsize=15)
        plt.xlabel("Time steps", fontsize=15, labelpad=-1)
    elif data_type == 'dqn':
        plt.ylabel("Max Q-value", fontsize=15)
        plt.xlabel("Episodes", fontsize=15, labelpad=-1)

    # our x-axis is iteration number

    # our task is called "Awesome Robot Performance"
    # plt.title("Awesome Robot Performance", fontsize=30)
    # Legend
    if action_type == 'discrete':
        plt.legend(labels=['bins = 5', 'bins = 7', 'bins = 11'], loc='lower right', markerscale=12, fontsize=15)
    elif action_type == 'continuous':
        plt.legend(labels=['continuous action'], loc='lower right', markerscale=12, fontsize=15)
    elif action_type == 'continuous combined':
        plt.legend(labels=['feature', 'feature + detection', 'pixels'], loc='lower right', markerscale=12, fontsize=15)

    # Show the plot on the screen
    fig.tight_layout()
    file_name = file_list[0].split('.')[-2].split('/')[-1]
    print("Save file at: " + "./pic/" + file_name + "_" + args.model +  ".png")
    if action_type != 'continuous-combined':
        plt.savefig("./pic/" + file_name + "_" + args.model +  ".png")
    else:
        plt.savefig("./pic/" + file_name + "_combined.png")
    plt.show()



if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="specify the model type")
    parser.add_argument("-a", "--action_type", help="specify the action type")

    args = parser.parse_args()
    data_type = 'episodic' # 'avg' or 'episodic'
    action_type = args.action_type # discrete, continuous or continuous combined

    if args.model is None or args.action_type is None:
        print("Model or action not specified. ")
        quit()



    folder_dir = 'data/'

    # file_list = ['franka-feature_len2048_l4.csv', 'franka-feature_len2048_l5.csv', 'franka-feature_len2048_l6.csv']
    # file_list = ['feature-n-detector_len128_l5.csv', 'feature-n-detector_len128_l7.csv', 'feature-n-detector_len128_l11.csv']
    # file_list = ['pixel_l5_scores.csv', 'pixel_l7_scores.csv', 'pixel_l11_scores.csv']
    # file_list = ['old-reacher/' + x for x in file_list]
    file_list = ['franka-pixel.csv']
    file_list = [os.path.join(folder_dir, args.model, file) for file in file_list]
    plot_data(file_list, data_type, action_type, head=50)


    #
    # if args.model == 'ppo_continuous':
    #
    #     # ppo continuous
    #     folder_dir = 'data/ppo_continuous/'
    #     # file_list = ['franka-pixel_len2048.csv']
    #     file_list = ['franka-feature_len2048.csv', 'franka-detector_len2048.csv']
    #     # file_list = ['feature_len128.csv', 'feature-n-detector_len128.csv', 'pixel_len128.csv']
    #     file_list = [folder_dir + x for x in file_list]
    #     plot_data(file_list, data_type, 'continuous')
    #
    # elif args.model == 'ppo_continuous_test':
    #     # ppo continuous
    #     folder_dir = 'data/ppo_continuous_test/'
    #     file_list = ['franka-feature.csv']
    #     file_list = [folder_dir + x for x in file_list]
    #     plot_data(file_list, data_type, 'continuous')
    #
    # elif args.model == 'ppo_discrete':
    #     # ppo discrete
    #     folder_dir = 'data/ppo_discrete/'
    #     # file_list = ['feature_l5.csv','feature_l7.csv','feature_l11.csv']
    #     # file_list = ['feature-n-detector_l5.csv','feature-n-detector_l7.csv','feature-n-detector_l11.csv']
    #     # observation = 'feature-n-detector'
    #     # discrete_level = ['5', '7', '11']
    #     # len = '128'
    #     # file_list = [ observation + '_len' + len + '_l' + d + '.csv' for d in discrete_level]
    #     # print("file_list: ", file_list)
    #     # file_list = ['franka-detector_len2048_l5.csv', 'franka-detector_len2048_l7.csv', 'franka-detector_len2048_l11.csv']
    #     # file_list = ['pixel_len128_l5.csv', 'pixel_len128_l7.csv', 'pixel_len128_l11.csv']
    #     file_list = ['franka-feature_len2048_l5.csv']
    #     file_list = [folder_dir + x for x in file_list]
    #     plot_data(file_list, data_type, 'discrete')
    #
    #
    # elif args.model == 'dqn_discrete':
    #     # dqn
    #     folder_dir = 'data/dqn_discrete/'
    #     # file_list = ['feature-n-detector_l5_scores.csv', 'feature-n-detector_l7_scores.csv', 'feature-n-detector_l11_scores.csv']
    #     # file_list = ['feature_l5_scores.csv', 'feature_l7_scores.csv', 'feature_l11_scores.csv']
    #     file_list = ['pixel_l5_scores.csv']
    #     file_list = [folder_dir + x for x in file_list]
    #     plot_data(file_list, 'dqn', 'discrete')
    #
    # elif args.model == 'dqn_test':
    #     # dqn
    #     folder_dir = 'data/dqn_test/old_data/'
    #     # file_list = ['feature_l5_q-mean.csv']
    #     file_list = ['feature_l5.csv','feature_l7.csv','feature_l11.csv']
    #     # file_list = ['feature_l5.csv', 'feature_l7.csv', 'feature_l11.csv']
    #     file_list = [folder_dir + x for x in file_list]
    #     plot_data(file_list, 'dqn', 'discrete')
    #
    # elif args.model == 'ddpg_continuous':
    #     # ddpg
    #     folder_dir = 'data/ddpg_continuous/'
    #     # file_list = ['feature.csv', 'feature-n-detector.csv','pixel.csv']
    #     file_list = ['franka-pixel.csv']
    #     file_list = [folder_dir + x for x in file_list]
    #     plot_data(file_list, data_type, 'continuous')
    #     # plot_data(file_list, data_type, 'continuous combined')
    #
    # elif args.model == 'ddpg_test':
    #     # ddpg
    #     folder_dir = 'data/ddpg_continuous_test/'
    #     file_list = ['feature.csv']
    #     file_list = [folder_dir + x for x in file_list]
    #     plot_data(file_list, data_type, 'continuous')
    #
    # elif args.model == 'ppo_discrete_test':
    #     # ppo discrete
    #     folder_dir = 'data/ppo_discrete_test/'
    #     file_list = ['franka-feature_len2048_l5.csv']
    #     file_list = [folder_dir + x for x in file_list]
    #     plot_data(file_list, data_type, 'discrete')
    # else:
    #     print("Model not specified.")
