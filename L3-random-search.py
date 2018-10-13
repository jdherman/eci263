import numpy as np 
import matplotlib.pyplot as plt

# function to optimize
# assume input x is a numpy array, not a list
def ackley(x):
  a = 20
  b = 0.2
  c = 2*np.pi
  d = x.size
  term1 = -a*np.exp(-b*np.sqrt((x**2).sum()/d))
  term2 = np.exp(np.cos(c*x).sum()/d)
  return (term1 - term2 + a + np.exp(1))

ub = 32.768
lb = -32.768

d = 2 # dimension of decision variable space
num_seeds = 10
max_NFE = 100000 # this is a lot, it will take a while
ft = np.zeros((num_seeds, max_NFE))

# pure random search
for seed in range(num_seeds):

  np.random.seed(seed)
  bestx = None
  bestf = None

  for i in range(max_NFE):

    x = np.random.uniform(lb, ub, d)
    f = ackley(x)

    if bestf is None or f < bestf:
      bestx = x
      bestf = f
      
    ft[seed,i] = bestf

  # for each trial print the result (but the traces are saved in ft)
  print(bestx)
  print(bestf)
  

plt.loglog(ft.T, color='steelblue', linewidth=1)
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.show()

