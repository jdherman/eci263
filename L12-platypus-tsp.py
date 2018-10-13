import numpy as np
from platypus import GeneticAlgorithm,Problem,Permutation,Insertion

def tour_distance(tour):
  # index with list, and repeat the first city
  tour = tour[0] # hack
  x = np.append(tour, tour[0])
  d = np.diff(xy[tour], axis=0) 
  return [np.sqrt((d**2).sum(axis=1)/10).sum()]

# load the data for the USA 48 capital cities
# optimal value = 10628 (https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html)
xy = np.loadtxt('data/tsp-48.txt')
num_cities = len(xy)

# Problem(number of decisions, number of objectives)
problem = Problem(1, 1)
problem.types[:] = Permutation(range(num_cities))
problem.directions[:] = Problem.MINIMIZE
problem.function = tour_distance

algorithm = GeneticAlgorithm(problem, variator = Insertion(probability=1.0))

# optimize the problem using 100000 function evaluations
# this doesn't seem to be working very well ...
# maybe a different operator is needed?
algorithm.run(100000)

# algorithm.result contains the whole population, so we 
# can grab the best solution
variables = np.array([s.variables for s in algorithm.result])
objectives = np.array([s.objectives for s in algorithm.result])

best_f = objectives.max()
best_x = variables[objectives.argmax()]

print(best_x)
print(best_f)