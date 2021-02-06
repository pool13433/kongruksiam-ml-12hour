import numpy as np
import matplotlib.pyplot as plt
# y = ax+b

x = np.linspace(-5, 5, 100)
y = 2*x+1
# print(y)

plt.plot(x, y, '-r', label='y =2x+1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="upper left")
plt.title('Graph y=2x+1')
plt.grid()
plt.show()
