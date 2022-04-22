from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
list(iris.keys())

X = iris["data"][:, (2,3)] #petal length, petal width
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)

softmax_reg.fit(X, y)

print(softmax_reg.predict([[5,2]]))
print(softmax_reg.predict_proba([[5,2]]))
