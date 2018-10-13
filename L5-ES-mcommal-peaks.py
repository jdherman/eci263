import numpy as np 
import matplotlib.pyplot as plt

# function to optimize
def peaks(x):
  a = 3*(1-x[0])**2*np.exp(-(x[0]**2) - (x[1]+1)**2)
  b = 10*(x[0]/5 - x[0]**3 - x[1]**5)*np.exp(-x[0]**2-x[1]**2)
  c = (1/3)*np.exp(-(x[0]+1)**2 - x[1]**2)
  return a - b - c + 6.5511 # add this so objective is always positive

ub = 3.0
lb = -3.0

d = 2 # dimension of decision variable space
s = 0.1 # stdev of normal noise
max_NFE = 10000

m = 10
l = 50 # beware "lambda" is a reserved keyword
num_seeds = 10

# empty matrix of objective values to fill in
ft = np.zeros((num_seeds, int(max_NFE/l)))

# separate function for mutation
def mutate(x, lb, ub, sigma):
  return np.clip(x + np.random.normal(0,sigma,len(x)), lb, ub)

# (mu,lambda) evolution strategy
for seed in range(num_seeds):
  np.random.seed(seed)

  # random initial population (l x d matrix)
  P = np.random.uniform(lb, ub, (l,d))
  f = np.zeros(l) # we'll evaluate them later
  nfe = 0
  f_best, x_best = None, None

  while nfe + l <= max_NFE:

    # evaluate all solutions in the population
    for i,x in enumerate(P):
      f[i] = peaks(x)
      nfe += 1

    # find m best parents, truncation selection
    ix = np.argsort(f)[:m]
    Q = P[ix, :] # parents

    # keep track of best here
    if f_best is None or f[ix[0]] < f_best:
      f_best = f[ix[0]]
      x_best = Q[0,:]

    # then mutate: each parent generates l/m children (integer division)
    child = 0
    for x in Q:
      for _ in range(int(l/m)):
        P[child,:] = mutate(x, lb, ub, s) # new population members
        child += 1

    ft[seed, int(nfe/l)-1] = f_best

  # for each trial print the result (but the traces are saved in ft)
  print(x_best)
  print(f_best)

nfe = range(l,max_NFE+1,l)
plt.loglog(nfe, ft.T, color='steelblue', linewidth=1)
plt.xlabel('NFE')
plt.ylabel('Objective Value')
plt.show()
