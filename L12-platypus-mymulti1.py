import numpy as np 
import matplotlib.pyplot as plt
from platypus import NSGAII, Problem, Real

# function to optimize
# from http://www.mathworks.com/help/gads/using-gamultiobj.html?refresh=true
def mymulti1(x):
  f1 = x[0]**4 - 10*x[0]**2+x[0]*x[1] + x[1]**4 -(x[0]**2)*(x[1]**2);
  f2 = x[1]**4 - (x[0]**2)*(x[1]**2) + x[0]**4 + x[0]*x[1];
  return [f1,f2]

# Problem(number of decisions, number of objectives)
problem = Problem(2, 2)
problem.types[:] = Real(-5, 5) # decision var. bounds
problem.function = mymulti1

algorithm = NSGAII(problem)

# examples for other algorithms (need to import separately):
# algorithm = NSGAII(problem)
# algorithm = NSGAIII(problem, divisions_outer=24)
# algorithm = CMAES(problem)
# algorithm = GDE3(problem)
# algorithm = IBEA(problem)
# algorithm = MOEAD(problem)
# algorithm = OMOPSO(problem, epsilons=[1.0,1.0])
# algorithm = SMPSO(problem)
# algorithm = SPEA2(problem)
# algorithm = EpsMOEA(problem, epsilons=[1.0,1.0])

# optimize the problem using 10000 function evaluations
algorithm.run(10000)

# either print the objectives ...
for solution in algorithm.result:
   print(solution.objectives)

# or plot them the same way we've been doing in class
# convert to numpy first
obj = np.array([s.objectives for s in algorithm.result])
x = np.array([s.variables for s in algorithm.result])

plt.subplot(1,2,1)
plt.scatter(x[:,0],x[:,1])
plt.ylim([-2,0])
plt.xlim([0,3])
plt.title('Decision space')

plt.subplot(1,2,2)
plt.scatter(obj[:,0],obj[:,1])
plt.ylim([0,35])
plt.xlim([-40,-5])
plt.title('Objective space')

plt.show()



