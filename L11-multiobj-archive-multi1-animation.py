import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from matplotlib import animation

# function to optimize
# from 
# http://www.mathworks.com/help/gads/using-gamultiobj.html?refresh=true
def mymulti1(x):
  f1 = x[0]**4 - 10*x[0]**2+x[0]*x[1] + x[1]**4 -(x[0]**2)*(x[1]**2);
  f2 = x[1]**4 - (x[0]**2)*(x[1]**2) + x[0]**4 + x[0]*x[1];
  return np.array([f1,f2])

ub = 5
lb = -5

d = 2 # dimension of decision variable space
num_obj = 2
s = 0.2 # stdev of normal noise (if this is too big, it's just random search!)

m = 18
l = 100 # beware "lambda" is a reserved keyword
max_gen = 200 # should be a multiple

# ASSUMES MINIMIZATION
# a dominates b if it is <= in all objectives and < in at least one
def dominates(a,b):
  return (np.all(a <= b) and np.any(a < b))

# select 1 parent from population P
# (Luke Algorithm 99 p.138)
def binary_tournament(P,f):
  ix = np.random.randint(0,P.shape[0],2)
  a,b = f[ix[0]], f[ix[1]]
  if dominates(a,b):
    return P[ix[0]]
  elif dominates(b,a):
    return P[ix[1]]
  else:
    return P[ix[0]] if np.random.rand() < 0.5 else P[ix[1]]

def mutate(x, lb, ub, sigma):
  return np.clip(x + np.random.normal(0,s,d), lb, ub)

# assumes minimization
# return only the nondominated members of A+P
# solution by Mohamed Alkaoud, Fall 2016
def archive_sort(A, fA, P, fP):

  A = np.concatenate((A, P))
  fA = np.concatenate((fA, fP))
  num_solutions = A.shape[0]

  # use a boolean index to keep track of nondominated solns
  keep = np.ones(num_solutions, dtype = bool)

  for i in range(num_solutions):
    keep[i] = np.all(np.any(fA >= fA[i], axis=1))

  return (A[keep], fA[keep])



# a simple multiobjective version of ES (sort of)
np.random.seed(2)

# random initial population (l x d matrix)
P = np.random.uniform(lb, ub, (l,d))
f = np.zeros((l,num_obj)) # we'll evaluate them later
gen = 0
P_save = []
f_save = []
A_save = []
Af_save = []

# archive starts with 2 bad made-up solutions
A = np.zeros_like(P[0:2,:])
fA = 10**10*np.ones_like(f[0:2,:])

while gen < max_gen:

  # evaluate all solutions in the population
  for i,x in enumerate(P):
    f[i,:] = mymulti1(x)

  A,fA = archive_sort(A, fA, P, f)

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
  A_save.append(np.copy(A))
  Af_save.append(np.copy(fA))
  gen += 1
  print(gen)

# Plotting stuff
plt.rcParams['figure.figsize'] = (8, 4)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Source Sans Pro'
plt.rcParams['axes.labelsize'] = 1.2*plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = 1.2*plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']


fig, ax = plt.subplots(1,2,figsize=(8,4))

def animate(i):
  ax[0].clear()
  X1,X2 = np.meshgrid(np.arange(0,3,0.01), np.arange(-2,0,0.01))
  # Z = mymulti1([X1,X2])
  # ax[0].contour(X1,X2,Z[0],50,cmap=plt.cm.Blues_r)
  # ax[0].contour(X1,X2,Z[1],50,cmap=plt.cm.Reds_r)
  ax[0].scatter(P_save[i][:,0], P_save[i][:,1], alpha=0.5, zorder=10)
  ax[0].scatter(A_save[i][:,0], A_save[i][:,1], color='indianred', alpha=0.5, zorder=10)
  ax[0].set_ylim([-2,0])
  ax[0].set_xlim([0,3])
  ax[0].set_title('Decision Space')

  ax[1].clear()
  ax[1].scatter(f_save[i][:,0], f_save[i][:,1], alpha=0.5)
  ax[1].scatter(Af_save[i][:,0], Af_save[i][:,1], color='indianred', alpha=0.5)
  ax[1].set_xlim([-40,-5])
  ax[1].set_ylim([0,35])
  ax[1].set_title('Objective Space, Generation ' + str(i))
  print(i)
  return None

anim = animation.FuncAnimation(fig, animate, frames=len(P_save))
anim.save('mymulti1-wcontour.gif', writer='imagemagick')
# animate(99)
plt.show()






