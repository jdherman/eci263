import numpy as np 
import matplotlib.pyplot as plt

from platypus.algorithms import *
from platypus.problems import DTLZ2
from platypus.indicators import Hypervolume

# from Dave Hadka:
# https://gist.github.com/dhadka/ba6d3c570400bdb411c3
# import random
# random.seed(1)

problem = DTLZ2(3) # 3-objective DTLZ2 problem

# setup the comparison
algorithms = [NSGAII(problem),
              NSGAIII(problem, divisions_outer=12),
              CMAES(problem, epsilons=[0.05]),
              GDE3(problem),
              IBEA(problem),
              MOEAD(problem),
              OMOPSO(problem, epsilons=[0.05]),
              SMPSO(problem),
              SPEA2(problem),
              EpsMOEA(problem, epsilons=[0.05])]

# run each algorithm for 10,000 function evaluations
for A in algorithms:
  print(A)
  A.run(1000)

# compute and print the hypervolume
# if we had a reference set, we'd use it here
hyp = Hypervolume(minimum=[0,0,0], maximum=[1,1,1])

for A in algorithms:
    print("%s\t%0.3f" % (A.__class__.__name__, hyp(A.result)))

