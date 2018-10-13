import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import rosen, differential_evolution
from scipy.interpolate import Rbf

# do this to ignore scipy RBF ill-conditioned matrix warnings
import warnings
warnings.filterwarnings("ignore")

# surrogate optimization on 2D rosenbrock function
def lhs(N,D):
  grid = np.linspace(0,1,N+1)
  result = np.random.uniform(low=grid[:-1], high=grid[1:], size=(D,N))
  for c in result:
    np.random.shuffle(c)
  return result.T

# wrapper around surrogate function, because it requires variables as 
# separate arguments (scipy gives a vector)
def stupid_wrapper(X):
  return s(*X)

bounds = [(-3,3)]*2
max_NFE = 3000
N_initial = 100

for seed in range(10):

  # initial sample and surrogate model
  X = lhs(N_initial, 2)*6-3 # scale up by bounds
  Z = np.array([rosen(z) for z in X])
  s = Rbf(X[:,0], X[:,1], Z, function='gaussian')

  nfe = N_initial
  bestf = np.min(Z)
  bestx = X[np.argmin(Z),:]

  while nfe < max_NFE and bestf > 10**-6:
    
    # optimize surrogate function to find next point
    result = differential_evolution(stupid_wrapper, bounds, polish=False)

    # add the new point to the set of "true" evaluations
    X = np.vstack((X, result.x))
    truef = rosen(result.x)
    Z = np.append(Z, truef)

    # fit a new surrogate model with the updated information
    s = Rbf(X[:,0], X[:,1], Z, function='gaussian')

    if truef < bestf:
      bestf = truef
      bestx = result.x

    nfe += 1

  print('Seed ' + str(seed) + ', NFE-to-converge: ' + str(nfe) + ', solution: ' + str(bestx))
