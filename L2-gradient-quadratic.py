import numpy as np 
import matplotlib.pyplot as plt

# function
def f(x):
  return 2*x[0]**2 + 10*x[1]**2

# gradient (normalized)
def fp(x):
  fx1 = 4*x[0]
  fx2 = 20*x[1]
  return np.array([fx1, fx2])

alpha = 0.01 # step size
x = np.array([3,3]) # initial point
xt = [] # empty lists to keep track of points we search
ft = []

# this is a "max-time" approach ... 
# could also loop until the gradient is within a tolerance
normgrad = 9999
i = 0
maxiter = 1000

while normgrad > 10**-8 and i < maxiter:
  xt.append(x)
  ft.append(f(x))
  grad = fp(x)
  x = x - alpha*grad
  normgrad = np.sqrt(np.sum(grad**2))
  i += 1

print(i)

# just plotting stuff below here ...
plt.subplot(1,2,1)
xx = np.arange(-5,5,0.005)
X1,X2 = np.meshgrid(xx, xx)
Z = 2*X1**2 + 10*X2**2
plt.contour(X1,X2,Z,50,cmap=plt.cm.Blues_r)
xt = np.array(xt)
plt.plot(xt[:,0], xt[:,1], color='k', linewidth=2)
plt.xlabel('X1')
plt.ylabel('X2')

plt.subplot(1,2,2)
plt.plot(ft, color='k', linewidth=2)
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.show()
