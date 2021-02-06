import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv(
    'https://raw.githubusercontent.com/kongruksiamza/MachineLearning/master/Linear%20Regression/Weather.csv')
# print(dataset.shape)
# print(dataset.describe())
"""
dataset.plot(x="MinTemp", y="MaxTemp", style="o")
plt.title('Min & Max Temp')
plt.xlabel('Min Temp')
plt.ylabel('Max Temp')
plt.show()
"""

#

x = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)

# 80%  20%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# training algrolithym
model = LinearRegression()
model.fit(x_train, y_train)

# test model  :: # y = ax+b
y_predict = model.predict(x_test)


# xxx
'''plt.scatter(x_test, y_test)
plt.plot(x_test, y_predict, color="red", linewidth=2)
plt.show()'''


# compare 2 data predict data
df = pd.DataFrame({'Actually': y_test.flatten(),
                   'Predicted': y_predict.flatten()})

# print(df.head())
# print(df.shape)
'''df1 = df.head(20)
df1.plot(kind="bar", figsize=(16, 10))
plt.show()
'''

print("MAE = ", metrics.mean_absolute_error(y_test, y_predict))
print("MSE = ", metrics.mean_squared_error(y_test, y_predict))
print("RMSE = ", np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
print("Score = ", metrics.r2_score(y_test, y_predict))
