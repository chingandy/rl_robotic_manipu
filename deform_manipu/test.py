# import os
# path = './modelpic/'
# if os.path.isdir(path):
#     print("exit")
# else:
#     os.makedirs(path)
from collections import deque
rewards = deque(maxlen=3)
rewards.append(3)
rewards.append(4)
rewards.append(2)
rewards.append(1)
print(rewards)
rewards.clear()
print(rewards)
