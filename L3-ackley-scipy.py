from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

x0 = [3,3]
bounds = [(lb,ub) for i in range(2)]
res = minimize(ackley, x0, bounds=bounds)
  
print(res)


