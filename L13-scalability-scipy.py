import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import rosen, differential_evolution, minimize
from scipy import stats

# converges to a very small tolerance, something like 10^-19
# num_seeds = 10

# for d in range(2,21): # loop over problem dimension
#   for seed in range(num_seeds):
#     np.random.seed(seed)
#     bounds = [(0,2)]*d
#     result = differential_evolution(rosen, bounds, polish=False)
#     # result = minimize(rosen, x0=np.random.uniform(0,2,d), bounds=bounds)
#     print('%d %d %d' % (d,seed,result.nfev))


# after running all of that and saving it to text files...

A = np.loadtxt('data/de-scaling-results.txt')
# A = np.loadtxt('data/bfgs-scaling-results.txt')

x = np.log(A[:,0])
y = np.log(A[:,2])
plt.scatter(x,y)

slope,intercept,rvalue,pvalue,stderr = stats.linregress(x,y)

plt.plot([np.min(x), np.max(x)], [intercept + np.min(x)*slope, intercept + np.max(x)*slope], color='indianred', linewidth=2)
plt.text(1.0,10, 'slope = %0.2f' % slope, fontsize=16)

plt.xlabel('# Decision Variables')
plt.ylabel('NFE to converge')
plt.title('DE - Rosenbrock function')
plt.show()