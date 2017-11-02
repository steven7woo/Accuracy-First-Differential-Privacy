#!/usr/bin/python3

import sklearn.datasets

# parameters are n, dimension, useful_dimension, noise level
n = 10**4
dim = 30
useful_dim = 25
sigma = 1.0

X, Y = sklearn.datasets.make_regression(n, dim, useful_dim, noise=sigma)

# maximum one-norm of any x or y
max_y_norm = max(Y)
max_x_norm = max([sum(list(map(abs,x))) for x in X])

for i in range(len(X)):
  xmap = X[i] / max_x_norm
  ymap = Y[i] / max_y_norm
  print(" ".join(list(map(str,xmap))), ymap)

