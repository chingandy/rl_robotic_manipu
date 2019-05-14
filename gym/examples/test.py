def my_function():
  """Docstring for my function
  you know nothing
  """
  #print the Docstring here.
  # print (my_function.__doc__)

print(my_function.__doc__)




# import logging
# logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
# logging.debug('some debugging details.')


# # def mySqrt(x: int) -> int:
# #     if x == 0:
# #         return 0
# #     for i in range(x+1):
# #         if i**2 == x:
# #             return i
# #         elif i**2 > x:
# #             return i - 1
# #         else:
# #             pass
#
# def mySqrt(x):
#     if x == 0:
#         return 0
#     sum = 0
#     for n in range(x):
#         sum += 2*n + 1
#         if sum == x:
#             return n + 1
#         elif sum > x:
#             return n
#         else:
#             pass
#
# print(mySqrt(
# 830674251))
# import numpy as np
# print("In test.py")
# v1 = np.array([1,1])
# v2 = np.array([0,0])
# vec = np.linalg.norm(v1-v2)
# print(vec)
# from collections import deque
# class Test:
#     def __init__(self, value):
#         self.val = value
#         self.rewards = deque(maxlen=3)
#     def get_value(self):
#         print("Value: ", self.val)
#     def get_rewards(self):
#         print(self.rewards)
#
# a = Test(3333)
# a.rewards.append(1234)
# a.get_value()
# a.get_rewards()
