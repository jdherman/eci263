import numpy as np 
import matplotlib.pyplot as plt

# rosenbrock function
# from http://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/optimize.html
def rosenbrock(x):
  return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

ub = 3.0
lb = -3.0

d = 2 # dimension of decision variable space
num_seeds = 30

popsize = 10
CR = 0.9 # crossover probability
F = 0.9 # between 0 and 2, vector step
max_NFE = 10000 # should be a multiple
convergence_threshold = 10**-6
nfe_to_converge = []

# differential evolution (a simple version)
for seed in range(num_seeds):
  np.random.seed(seed)

  # random initial population (popsize x d matrix)
  P = np.random.uniform(lb, ub, (popsize,d))
  f = np.zeros(popsize) # we'll evaluate them later
  nfe = 0
  f_best, x_best = 9999999, None
  ft = []

  while nfe < max_NFE and f_best > convergence_threshold:

    # for each member of the population ..
    for i,x in enumerate(P):
      
      # pick two random population members
      # "x" will be the one we modify, but other variants
      # will always modify the current best solution instead
      xb,xc = P[np.random.randint(0, popsize, 2), :]
      v = x + F*(xb-xc) # mutant vector

      # crossover: either choose from x or v
      trial_x = np.copy(x)
      for j in range(d):
        if np.random.rand() < CR:
          trial_x[j] = v[j]

      f[i] = rosenbrock(x)
      trial_f = rosenbrock(trial_x)
      nfe += 1

      # if this is better than the parent, replace
      if trial_f < f[i]:
        P[i,:] = trial_x
        f[i] = trial_f

    # keep track of best here
    if f_best is None or f.min() < f_best:
      f_best = f.min()
      x_best = P[f.argmin(),:]

    ft.append(f_best)

  nfe_to_converge.append(nfe)
  fta = np.array(ft)
  plt.subplot(1,2,1)
  plt.semilogy(range(popsize,nfe+1,popsize), fta, color='steelblue', linewidth=1)
  plt.subplot(1,2,2)
  plt.plot(range(2*popsize,nfe+1,popsize), fta[1:]/fta[:-1], color='steelblue', linewidth=1)
  del ft[:]


print(nfe_to_converge)

plt.subplot(1,2,1)
plt.xlabel('NFE')
plt.ylabel('Objective Value')

plt.subplot(1,2,2)
plt.xlabel('NFE')
plt.ylabel('$f_{t} / f_{t-1}$')
plt.show()






