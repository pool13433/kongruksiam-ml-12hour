import numpy as np
import matplotlib.pyplot as plt

rng = np.random
x = rng.rand(50)*10
y = 2*x+rng.randn(50)
print(y)

#plt.plot(x, y)
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
