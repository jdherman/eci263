import numpy as np 
import matplotlib.pyplot as plt

# example: Latin Hypercube sampling
def lhs(N,D):
  grid = np.linspace(0,1,N+1)
  result = np.random.uniform(low=grid[:-1], high=grid[1:], size=(D,N))
  for c in result:
    np.random.shuffle(c)

  return result.T

N = 100
X = lhs(N, 2)
Y = np.random.uniform(0,1,(N,2))

# First subplot: LHS
plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1])
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('LHS')
if N < 20:
  plt.gca().set_xticks(np.linspace(0,1,N+1))
  plt.gca().set_yticks(np.linspace(0,1,N+1))

# Second subplot: Uniform random
plt.subplot(1,2,2)
plt.scatter(Y[:,0], Y[:,1])
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('Random')
if N < 20:
  plt.gca().set_xticks(np.linspace(0,1,N+1))
  plt.gca().set_yticks(np.linspace(0,1,N+1))

plt.show()