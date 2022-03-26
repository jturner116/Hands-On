import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = 2 * np.random.rand(100, 1)
y = 4 + 3*X + np.random.rand(100,1)

X_b = np.c_[np.ones((100,1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

lin_reg = LinearRegression()
lin_reg.fit(X, y)

print(lin_reg.intercept_)
print(lin_reg.coef_)

plt.plot(X, y, 'ro')
plt.plot(X, lin_reg.predict(X))
plt.show()



