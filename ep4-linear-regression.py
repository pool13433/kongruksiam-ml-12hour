import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rng = np.random
x = rng.rand(50)*10
y = 2*x+rng.randn(50)

# linear regression model
model = LinearRegression()

# reshape
x_new = x.reshape(-1, 1)
# print(x_new)
y_new = y.reshape(-1, 1)
# print(y_new)

# print(model.score(x_new, y))
# print(model.intercept_)
# print(model.coef_)


# train algorithym
model.fit(x_new, y)

# test model
xfit = np.linspace(-1, 11)
xfit_new = xfit.reshape(-1, 1)
# print(xfit_new.shape)

# test model
yfit = model.predict(xfit_new)

# analysis
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()
