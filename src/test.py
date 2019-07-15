<<<<<<< HEAD
def test(x):
    print("in the test function")
    return x + 3
x = lambda: test()

num = 10
print(x(3))



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
=======
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()
>>>>>>> 812e1d69e1785d5425dfefc4e6ed25959f53799d
