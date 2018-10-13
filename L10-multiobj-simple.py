import numpy as np 
import matplotlib.pyplot as plt

# function to optimize
def schaffer(x):
  f1 = x[0]**2
  f2 = (x[0]-2)**2
  return np.array([f1,f2])

# from 
# http://www.mathworks.com/help/gads/using-gamultiobj.html?refresh=true
# def mymulti1(x):
#   f1 = x[0]**4 - 10*x[0]**2+x[0]*x[1] + x[1]**4 -(x[0]**2)*(x[1]**2);
#   f2 = x[1]**4 - (x[0]**2)*(x[1]**2) + x[0]**4 + x[0]*x[1];
#   return np.array([f1,f2])

ub = 10
lb = -10

d = 1 # dimension of decision variable space
num_obj = 2
s = 0.1 # stdev of normal noise (if this is too big, it's just random search!)

m = 15
l = 100 # beware "lambda" is a reserved keyword
max_gen = 50 # should be a multiple

# ASSUMES MINIMIZATION
# a dominates b if it is <= in all objectives and < in at least one
def dominates(a,b):
  return (np.all(a <= b) and np.any(a < b))

# select 1 parent from population P
# (Luke Algorithm 99 p.138)
def binary_tournament(P,f):
  ix = np.random.randint(0,l,2)
  a,b = f[ix[0]], f[ix[1]]
  if dominates(a,b):
    return P[ix[0]]
  elif dominates(b,a):
    return P[ix[1]]
  else:
    return P[ix[0]] if np.random.rand() < 0.5 else P[ix[1]]

def mutate(x, lb, ub, sigma):
  return np.clip(x + np.random.normal(0,s,d), lb, ub)

# a simple multiobjective version of ES (sort of)
np.random.seed(1)

# random initial population (l x d matrix)
P = np.random.uniform(lb, ub, (l,d))
f = np.zeros((l,num_obj)) # we'll evaluate them later
gen = 0
P_save = []
f_save = []

while gen < max_gen:

  # evaluate all solutions in the population
  for i,x in enumerate(P):
    f[i,:] = schaffer(x)

  # find m parents from nondomination tournaments
  Q = np.zeros((m,d))
  for i in range(m):
    Q[i,:] = binary_tournament(P,f)

  # then mutate: each parent generates l/m children (integer division)
  child = 0
  for i,x in enumerate(Q):
    for j in range(int(l/m)):
      P[child,:] = mutate(x, lb, ub, s)
      child += 1

  P_save.append(np.copy(P))
  f_save.append(np.copy(f))
  gen += 1


plt.subplot(1,2,1)
plt.scatter(P_save[i][:,0], np.zeros(len(P_save[i][:,0])), alpha=0.5)
plt.ylim([-1,1])
plt.xlim([-5,5])
plt.title('Decision Space (one variable)')

plt.subplot(1,2,2)
plt.scatter(f_save[i][:,0], f_save[i][:,1], alpha=0.5)
plt.xlim([0,5])
plt.ylim([0,5])
plt.title('Objective Space, Generation ' + str(i))

plt.show()
