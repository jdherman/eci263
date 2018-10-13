import numpy as np 
import matplotlib.pyplot as plt

# 0-1 knapsack
def value(x, v, w, W):
  # if weight is exceeded, value=0
  if np.sum(x*w) > W:
    return 0
  else:
    return np.sum(v*x)

d = 10 # dimension of decision variable space
num_seeds = 20

popsize = 80
pc = 0.9
pm = 0.1 # recommended 1/D
max_gen = 50
k = 2 # tournament size
ft = np.zeros((num_seeds, max_gen))

# knapsack problem P01
# Optimal solution = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
# which has a weight of 165 and a value of 309 
W = 165
w = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82])
v = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72])

# select 1 parent from population P
# using tournaments of size k
def tournament_selection(P,f,k):
  candidates = np.random.randint(0,popsize,k)
  best = f[candidates].argmax()
  index = candidates[best]
  return P[index,:]

# one-point crossover plus mutation
# input two parents and crossover probabilities
def cx_and_mut(P1,P2,pc,pm):
  child1 = np.copy(P1)
  child2 = np.copy(P2)

  # one-point crossover
  if np.random.rand() < pc:
    x = np.random.randint(d)
    child1[x:] = P2[x:]
    child2[:x] = P1[:x]

  # bit-flip mutation
  for c in child1,child2:
    for i in range(d):
      if np.random.rand() < pm:
        c[i] = 1 - c[i]

  return child1,child2

# run the GA
for seed in range(num_seeds):
  np.random.seed(seed)

  # initialize
  P = np.random.randint(0, 2, (popsize,d))
  f = np.zeros(popsize) # we'll evaluate them later
  gen = 0
  f_best, x_best = None, None

  while gen < max_gen:

    # evaluate
    for i,x in enumerate(P):
      f[i] = value(x, v, w, W)

    # keep track of best
    if f_best is None or f.max() > f_best:
      f_best = f.max()
      x_best = P[f.argmax(),:]
      
    # selection / crossover / mutation (following Luke Algorithm 20)
    Q = np.copy(P)
    for i in range(0, popsize, 2):
      parent1 = tournament_selection(P,f,k)
      parent2 = tournament_selection(P,f,k)
      child1,child2 = cx_and_mut(parent1,parent2,pc,pm)
      Q[i,:] = child1
      Q[i+1,:] = child2

    # new population of children
    P = np.copy(Q)
    ft[seed,gen] = f_best
    gen += 1

  # for each trial print the result (but the traces are saved in ft)
  print(x_best)
  print(f_best)


plt.plot(ft.T, color='steelblue', linewidth=1)
plt.xlabel('Generations')
plt.ylabel('Objective Value')
plt.show()

