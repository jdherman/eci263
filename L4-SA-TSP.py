import numpy as np 
import matplotlib.pyplot as plt

# this data uses "ATT distance" defined here:
# http://debian.cse.msu.edu/CSE/pub/GA/programs/Genocop-doc/TSP/READ_ME
# (this explains the division by 10 inside the square root)
def distance(tour, xy):
  # index with list, and repeat the first city
  tour = np.append(tour, tour[0])
  d = np.diff(xy[tour], axis=0) 
  return np.sqrt((d**2).sum(axis=1)/10).sum()

num_seeds = 11
max_NFE = 100000
ft = np.zeros((num_seeds, max_NFE))
T0 = 1000 # initial temperature
alpha = 0.95 # cooling parameter

# load the data for the USA 48 capital cities
# optimal value = 10628 (https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html)
xy = np.loadtxt('data/tsp-48.txt')
num_cities = len(xy)

# simulated annealing
for seed in range(num_seeds):
  np.random.seed(seed)
  T = T0

  # random initial tour
  tour = np.arange(num_cities)
  tour = np.random.permutation(tour)
  bestf = distance(tour, xy)

  for i in range(max_NFE):

    # mutate the tour using two random cities
    trial_tour = np.copy(tour) # do not operate on original list
    
    # 2018: FIX to not allow same a,b
    a = np.random.randint(num_cities)
    b = np.random.randint(num_cities)

    # option 1: just swap the pair (in-class version)
    trial_tour[a],trial_tour[b] = trial_tour[b],trial_tour[a]

    trial_f = distance(trial_tour, xy)

    r = np.random.rand()
    if T > 10**-3: # protect division by zero
      P = np.min([1.0, np.exp((bestf - trial_f)/T)])
    else:
      P = 0.0
    
    if trial_f < bestf or r < P:
      tour = trial_tour
      bestf = trial_f
    
    T = T0*alpha**i
    ft[seed,i] = bestf

  print(tour)
  print(bestf)

plt.subplot(1,2,1)
tour = np.append(tour, tour[0]) # for plotting
plt.plot(xy[tour][:,0], xy[tour][:,1], marker='o')

plt.subplot(1,2,2)
plt.semilogx(ft.T, color='steelblue', linewidth=1)
plt.xlabel('Iterations')
plt.ylabel('Length of Tour')
plt.show()



