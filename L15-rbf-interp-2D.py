import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

def f(x):
  return x[0]*np.exp(-x[0]**2-x[1]**2)

# setup data
lb = -2
ub = 2
xx = np.linspace(lb,ub, 500)
X1,X2 = np.meshgrid(xx, xx)
truef = f([X1,X2])

# setup figure
fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
plt.xlabel('X')
plt.ylabel('Y')

# create the surrogate function using a random sample of x,y --> z
x,y = np.random.uniform(lb,ub, size=(2,20))
z = f([x,y])
s = Rbf(x,y,z, function='cubic')

# plot the true function
ax1.plot_surface(X1, X2, truef, cmap=plt.cm.jet, edgecolor='none')
ax1.set_title('True function $f(x)$')

# plot the surrogate function
ax2.plot_surface(X1, X2, s(X1,X2), cmap=plt.cm.jet, edgecolor='none')
ax2.set_title('Surrogate function $s(x)$')

# scatter plot the points used to create the surrogate function
ax2.scatter(x, y, -0.5, color='k', marker='o', s=40, alpha=1)

# some plot settings
for ax in ax1,ax2:
  ax.set_xlim([lb,ub])
  ax.set_ylim([lb,ub])
  ax.set_zlim([-.5,.5])
  ax.view_init(20,-60)

plt.show()