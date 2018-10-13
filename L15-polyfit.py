import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
beta = np.polyfit(x, y, deg=5)

# create a polynomial function from the beta coefficients
p = np.poly1d(beta)
# print(p)
xx = np.linspace(0.0,5.0, 100)

plt.scatter(x,y, 50, 'steelblue')
plt.plot(xx, p(xx), color='indianred', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
