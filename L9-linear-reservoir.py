import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# load historical data from file
data = np.loadtxt('data/leaf-river-data.txt', skiprows=1)
P = data[:,3]
ET = data[:,4]
Q_obs = data[:,5]
N = len(P)

def linear_reservoir(k, optimizing=True):
  Q = np.zeros(N)
  S = np.zeros(N)
  S[0] = 0 # initial condition

  for t in range(1,N):
    Q[t] = k*S[t-1]
    S[t] = S[t-1] + P[t] - ET[t] - Q[t]

    if S[t] < 0:
      S[t] = 0

  if optimizing:
    rmse = np.sqrt(((Q-Q_obs)**2).mean())
    return rmse
  else:
    return Q

# optimize with DE
result = differential_evolution(linear_reservoir, bounds=[(0,1)], polish=False)
best_k = result.x
print(result)

# re-run to get plot
Q = linear_reservoir(best_k, optimizing=False)
plt.plot(Q_obs, color='k')
plt.plot(Q, color='red')
plt.xlabel('Days')
plt.ylabel('Streamflow (mm)')
plt.legend(['Observed', 'Simulated'])
plt.show()
