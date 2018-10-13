import numpy as np 
import matplotlib.pyplot as plt

# reference:
# http://docs.scipy.org/doc/numpy/reference/routines.random.html

# specify random seed - same results every time
np.random.seed(1)

# generate 10 values between 0 and 1
x = np.random.rand(10)
# print(x) # confirm it's the same every time

# generate 1000 values uniformly distributed between 15 and 20
# x = np.random.uniform(15, 20, 10000)

# # generate 1000 normally distributed values
# # with mean 5 and standard deviation 2
x = np.random.normal(5, 2, 100000)

# # # check the distribution with a histogram
# plt.hist(x)
# plt.show()


# how about for discrete / combinatorial values?

# generate 10 random integers between 0 and 100
# x = np.random.randint(0, 100, 10)
# print(x)

# x = np.random.permutation(x)
# print(x)

# see also np.random.choice to 
# choose values from an existing array

