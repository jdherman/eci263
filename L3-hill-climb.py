import numpy as np 
import matplotlib.pyplot as plt

# function to optimize
# assume input x is a numpy array, not a list
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

d = 2 # dimension of decision variable space
s = 0.5 # stdev of normal noise (if this is too big, it's just random search!)
num_seeds = 10
max_NFE = 100000
ft = np.zeros((num_seeds, max_NFE))

# hill climbing
for seed in range(num_seeds):
  np.random.seed(seed)

  # random initial starting point
  x = np.random.uniform(lb, ub, d)
  bestf = ackley(x)

  for i in range(max_NFE):
    
    # normal distribution: don't allow points outside bounds
    # trial_x = x + np.random.normal(0,s,d)
    # while np.any((trial_x > ub) | (trial_x < lb)):
    #   trial_x = x + np.random.normal(0,s,d)

    # or use a truncated normal sample to avoid loop:
    trial_x = np.clip(x + np.random.normal(0,s,d), lb, ub)

    trial_f = ackley(trial_x)
    
    # only keep solutions that are better
    if trial_f < bestf:
      x = trial_x
      bestf = trial_f
    
    ft[seed,i] = bestf

  # for each trial print the result (but the traces are saved in ft)
  print(x)
  print(bestf)
  

plt.loglog(ft.T, color='steelblue', linewidth=1)
plt.ylim([10**-2,10**2])
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.show()



