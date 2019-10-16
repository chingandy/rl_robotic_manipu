import torch
import time

start_time = time.time()
x = torch.randn(1)

if torch.cuda.is_available():
    device = torch.device("cuda:4")          # a CUDA device object
    print("Device: ", device)
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
end_time = time.time()
print("Elapsed time = ", end_time - start_time)







# import csv
# import numpy as np
# file_dir = 'data/ppo_continuous/feature_128.csv'
# i = 0
# data = []
# with open(file_dir) as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#         row = [float(x) for x in row]
#         data.append(row)

# print(len(data[0]))

# save_dir = 'data/test.csv'
# with open(save_dir, mode='a') as log_file:
#     writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     # print("episodic returns: ", agent.episodic_returns)
#     writer.writerow([2,3,4,5])



# import os
# path_1 = 'chingandywu'
# path_2 = 'master-thesis/code'
#
# print(os.path.join(path_1, path_2))
#
#

# import torch
# x = torch.tensor([0.1, -0.3, 0.5, -.4])
# print(x)
# m = torch.nn.Softmax(dim=0)
# y = m(x)
# print(y)
#
# # array = [1]
# # n = np.array(array)
# # print(n)
# # if n:
# #     print("T")
# # else:
# #     print("F")
# # quit()
# l = []
# l = np.array(l)
# print(len(l))
# quit()







# value = 1.512
# print(np.subtract.outer(array, value))
# print(np.abs(np.subtract.outer(array, value)))
#
# indices = np.abs(np.subtract.outer(array, value)).argmin(0)
# print(indices)





# m = torch.distributions.Categorical(torch.Tensor([ 0.25, 0.25, 0.25, 0.25 ]))
# x = m.sample()
# log_prob = m.log_prob(x)
# print(x)
# print(log_prob)




# import csv
#
# input_1 = [1,2,3,4]
# input_2 = [4,5,6,7]
# # with open('test.csv', mode='a') as log_file:
# #     writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
# #     writer.writerow(input_1)
# #
# # with open('test.csv', mode='a') as log_file:
# #     writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
# #     writer.writerow(input_2)
#
#
# with open('test.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#         print("Row %d: " % line_count, row)
#         print(row[0], type(row[0]))
#
#         line_count += 1
#     print('Processed %d lines.' % line_count)


# car = {
#   "brand": "Ford",
#   "model": "Mustang",
#   "year": 1964
# }
# print(car)
#
# x = car.setdefault("color", "white")
# print(x)
# print(car)
#

# class Test:
#     def __init__(self, value):
#         self.x = value
#     # def __call__(self, y):
#     #     self.x = y
#     #     print("x: ", self.x)
#     #     return
#
# a = Test(100)
# print(a(10))

# def build_profile(first, last, **user_info):
#     """Build a dictionary containing everything we know about a user."""
#     profile = {}
#     profile['first_name'] = first
#     profile['last_name'] = last
#     for key, value in user_info.items():
#         profile[key] = value
#     return profile
#
# user_profile = build_profile('albert', 'eistein', location='princeton', field='physics')
#
# print(user_profile)




# import torch
# import numpy as np
# from torch.distributions import MultivariateNormal
#
# x = torch.tensor([1.0, 1.0, 1.0])
# y = torch.tensor([2.0, 2.0, 2.0])
#
# z = x / y
# print(z)



# a = torch.randn((2, 3, 4))
# b = torch.randn((2, 3, 4))
# # b = torch.unsqueeze(b, dim=2)  # 2, 3, 1
# # print(a.size())
# # print(b.size())
# # torch.unsqueeze(b, dim=-1) does the same thing
#
# x = torch.stack([a, b], dim=3)  # 2, 3, 5
# print(x.size())
# quit()
#
#
# x = torch.tensor([1.0,2.0,3.0])
# print(x.mean().item())
# quit()
#
#
# x = np.arange(10).reshape(2,5)
# print(x)
# x = x.flatten()
# print(x)
# quit()
#




# m = MultivariateNormal(torch.zeros(2), torch.eye(2))
# print(m.sample())
# x = torch.tensor([[1,2,3], [4,5,6]])
# print(x)
# x = x.view((1,2,3))
# print(x.shape)
# x = torch.squeeze(x)
# print(x.shape)

# x = torch.full((2, 3), 3.1425)
# print(x)
# =======
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Data for plotting
# t = np.arange(0.0, 2.0, 0.01)
# s = 1 + np.sin(2 * np.pi * t)
#
# fig, ax = plt.subplots()
# ax.plot(t, s)
#
# ax.set(xlabel='time (s)', ylabel='voltage (mV)',
#        title='About as simple as it gets, folks')
# ax.grid()
#
# fig.savefig("test.png")
# plt.show()
# >>>>>>> 812e1d69e1785d5425dfefc4e6ed25959f53799d
