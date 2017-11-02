# ridge regression
# solves min_beta  0.5 * (1/n) * sum_t (y_t - beta*x_t)^2  +  0.5*lamb*||beta||_2^2
#
# via the closed form:
#   Sigma = X^T . X + lamb*I
#   R = X^T . Y
#   beta = Sigma^-1 . R

import numpy as np
import math
from sklearn import linear_model

# our file
import project


# ---------------------------------------------

# Max norm of optimal hypothesis when this is the regularization param
def compute_max_norm(lamb):
  return 1.0 / math.sqrt(lamb)


# Sensitivity of a query to sparse vector
def get_sv_sensitivity(max_norm, n):
  return 0.5 * (max_norm+1.0)**2.0 / n


# L1-sensitivity of the optimal beta
def compute_opt_sensitivity(n, dim, lamb, max_norm):
  return (max_norm+1.0) * math.sqrt(dim / (n*lamb))


# ---------------------------------------------




# empirical risk of beta on the data set
# note: this method is a bottleneck,
# use the for loop to speed it up a bit
def compute_err(X, Y, lamb, beta):
  n = len(Y)
  total = 0.0
  for i in range(n):
    temp = np.dot(X[i],beta) - Y[i]
    total += temp*temp
  avg_loss = 0.5 * total / n
  two_norm = beta.dot(beta)
  return 0.5*(avg_loss + lamb * two_norm)


def compute_Sigma(X, lamb):
  return X.T.dot(X) + len(X) * lamb * np.identity(len(X[0]))


def compute_R(X, Y):
  return X.T.dot(Y)


# return optimal beta
def exact_solve(Sigma, R):
  return np.linalg.inv(Sigma).dot(R)


# solve exactly from data
# mainly used as a check that our code was right
def exact_solve_data(X, Y, lamb):
  reg = linear_model.Ridge(alpha = lamb * len(X), fit_intercept=False)
  reg.fit(X, Y)
  return reg.coef_


# return optimal beta projected to L2-norm max_norm
def exact_solve_and_project(Sigma, R, max_norm):
  beta = exact_solve(Sigma, R)
  return project.two_norm_project(beta, max_norm)




