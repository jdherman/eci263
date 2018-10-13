import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# function
def f(x):
  return 2*x[0]**2 + 10*x[1]**2

# gradient (normalized)
def fp(x):
  fx1 = 4*x[0]
  fx2 = 20*x[1]
  # mag = np.sqrt(fx1**2+fx2**2)
  return np.array([fx1, fx2])

x0 = [3,3]
res = minimize(f, x0)#, jac=fp)
  
print(res)
