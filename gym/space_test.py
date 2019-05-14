from gym import spaces
space = spaces.Discrete(8) # set with 8 elements {0, 1, 2, 3, 4, 5, 6, 7}
x = space.sample()
print(space)
print(x)
assert space.contains(x)
assert space.n == 8
