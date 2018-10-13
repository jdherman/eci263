import numpy as np 
import matplotlib.pyplot as plt

# function
def f(x):
  return x[0]**4 - 3*x[0]**3 + 4*x[0] + x[1]**4 - 3*x[1]**3 + 4*x[1]

# gradient (normalized)
def fp(x):
  fx1 = 4*x[0]**3 - 9*x[0]**2 + 4
  fx2 = 4*x[1]**3 - 9*x[1]**2 + 4
  return np.array([fx1, fx2])

alpha = 0.01 # step size
num_trials = 100
nfe = 200
xt = []
ft = []
initx = []

# multistart
for trial in range(num_trials):
  x = -2 + 5*np.random.rand(2,) # initial point
  initx.append(x)

  for i in range(nfe):
    xt.append(x)
    ft.append(f(x))
    x = x - alpha*fp(x)

# just plotting stuff below here ...
plt.subplot(1,2,1)
xx = np.arange(-2,3,0.01)
X1,X2 = np.meshgrid(xx, xx)
Z = f([X1,X2])
plt.contour(X1,X2,Z,100,cmap=plt.cm.Blues_r, zorder=-10)
xt = np.array(xt).reshape((num_trials,nfe,2))

for i in range(num_trials):
  plt.scatter(xt[i,0,0], xt[i,0,1], s=30, c='k')
  plt.plot(xt[i,:,0], xt[i,:,1], color='k', linewidth=1)

plt.xlabel('X1')
plt.ylabel('X2')

plt.subplot(1,2,2)
ft = np.array(ft).reshape((num_trials,nfe))
for i in range(num_trials):
  plt.plot(ft[i,:], color='k', linewidth=1)

plt.xlabel('Iterations')
plt.ylabel('Objective Value')


plt.show()
