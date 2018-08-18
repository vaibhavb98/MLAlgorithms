import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.datasets as skds
import sklearn.preprocessing as pp
from sklearn.model_selection import train_test_split

dataset = skds.load_boston()

df = pd.DataFrame(dataset['data'])
df.columns = dataset['feature_names']
X = df.values
mms = pp.MinMaxScaler()
X = mms.fit_transform(X)
# print(pd.DataFrame(X).head(3))

from math import floor


def LinearRegressor(X, y, inital_weight=0, initial_bias=0, n_batch=43, epochs=100, learning_rate=0.01):
    N, weight, bias, cost = float(len(y)), inital_weight, initial_bias, None
    x = X.reshape(n_batch, floor(len(X)/n_batch))
    for i in range(epochs):
        for Xi in X:
            y_pred = (weight * Xi) + bias
            diff = (y - y_pred)
            cost = sum([x**2 for x in diff]) / N
            weight_gradient = (-2 / N) * (sum([x for x in diff*Xi]))
            bias_gradient = (-2 / N) * (sum([x for x in diff]))
            weight -= learning_rate * weight_gradient
            bias -= learning_rate * bias_gradient
    return weight, bias, cost


X_train, x_test, y_train, y_test = train_test_split(
    X, dataset['target'] * 1000, test_size=0.15)

X_train, x_test = np.array([sum(x) for x in X_train]), np.array([
    sum(x) for x in x_test])

weight, bias, cost = LinearRegressor(X_train, y_train)

y_pred = weight * x_test + bias

plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color='red')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.show()

# from sklearn.metrics import r2_score
# print(r2_score(y_test, y_pred))
