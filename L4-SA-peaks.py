import numpy as np 
import matplotlib.pyplot as plt

# function to optimize
# assume input x is a numpy array, not a list
def peaks(x):
  a = 3*(1-x[0])**2*np.exp(-(x[0]**2) - (x[1]+1)**2)
  b = 10*(x[0]/5 - x[0]**3 - x[1]**5)*np.exp(-x[0]**2-x[1]**2)
  c = (1/3)*np.exp(-(x[0]+1)**2 - x[1]**2)
  return a - b - c + 6.551 # add this so objective is always positive

ub = 3.0
lb = -3.0

d = 2 # dimension of decision variable space
s = 0.5 # stdev of normal noise (if this is too big, it's just random search!)
num_seeds = 5
max_NFE = 100000
ft = np.zeros((num_seeds, max_NFE))
xt = np.zeros((2, max_NFE))
T0 = 100 # initial temperature
alpha = 0.95 # cooling parameter

# three different x's and f's we're keeping track of:
# x_current: the location right now, which can improve or worsen
# x_best: the best point found so far (can only improve)
# x_trial: a new candidate point which we may or may not accept

# simulated annealing
for seed in range(num_seeds):
  np.random.seed(seed)
  T = T0

  # random initial starting point
  x_current = np.random.uniform(lb, ub, d)
  f_current = peaks(x_current)
  x_best,f_best = x_current,f_current

  for i in range(max_NFE):

    # do not allow points outside the constraints
    x_trial = np.clip(x_current + np.random.normal(0,s,d), lb, ub)
    f_trial = peaks(x_trial)

    r = np.random.rand()
    if T > 10**-3: # protect division by zero
      P = np.min([1.0, np.exp((f_current - f_trial)/T)])
    else:
      P = 0.0
    
    # keep better solutions, and worse ones sometimes
    if f_trial < f_current or r < P:
      x_current = x_trial
      f_current = f_trial
    
    if f_trial < f_best:
      x_best = x_trial
      f_best = f_trial 
    
    T = T0*alpha**i
    ft[seed,i] = f_current
    xt[:,i] = x_current

  # for each trial print the result (but the traces are saved in ft)
  print(x_best)
  print(f_best)
  

plt.subplot(1,2,1)
xx = np.arange(-3,3,0.01)
X1,X2 = np.meshgrid(xx, xx)
Z = peaks([X1,X2])
plt.contour(X1,X2,Z,50,cmap=plt.cm.Blues_r)
# if you run multiple seeds, this will only plot the last trace of xt
plt.plot(xt[0,:], xt[1,:], color='k', linewidth=2)
plt.xlabel('X1')
plt.ylabel('X2')
plt.colorbar()

plt.subplot(1,2,2)
plt.loglog(ft.T, color='steelblue', linewidth=1)
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.show()



