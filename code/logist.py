# logistic regression

# minimize (1/n)sum_t ln(1 + e(-y_t beta.x_t)) + 0.5*lambda*||beta||_2^2

import math
import numpy as np
#from cvxpy import *
from sklearn.linear_model import LogisticRegression

MAX_ERR = 1000000000


# ---------------------------------------------

# Max norm of optimal hypothesis when this is the regularization param
def compute_max_norm(lamb):
  return math.sqrt(2.0*math.log(2.0) / lamb)


# Sensitivity of a query to sparse vector
def get_sv_sensitivity(max_norm, n):
  return math.log(1.0 + math.exp(max_norm)) / n


# L1-sensitivity of the optimal beta
def compute_opt_sensitivity(n, dim, lamb):
  return 2.0 * math.sqrt(dim) / (n*lamb)


# ---------------------------------------------

def compute_err(X, Y, lamb, beta):
  n = len(X)
  total = 0.0
  try:
    for i in range(n):
      total += math.log(1.0 + math.exp(-Y[i] * np.dot(X[i], beta)))
  except OverflowError:
    return MAX_ERR
  avg = total / n
  twonorm = np.dot(beta, beta)
  return avg + 0.5 * lamb * twonorm


# ---------------------------------------------

def logistic_regression(X, Y, lamb):
  C = 1.0 / (lamb * len(X))
  lr = LogisticRegression(penalty="l2", C=C, fit_intercept=False)
  lr.fit(X, Y)
  beta = np.array(lr.coef_[0])
  return beta, compute_err(X, Y, lamb, beta)


# Problem: this took too much memory (5+GB on a 120MB input data file)
# Input:
#   X and Y are numpy arrays,
#   X is dim by n
#   Y is 1 by n, each entry is {-1, +1}
#   lamb is regularization constant
#
# Output:
#   optimal hypothesis beta
#   value of its solution (optimal regularized error)
#def cvxpy_logistic_regression(X, Y, lamb):
#  n = len(Y)
#  d = X.shape[1]
#  beta = Variable(d)
#  # this version threw an error for me
#  #expr1 = sum([log_sum_exp(vstack(0, -Y[i] * np.dot(X[i, :], beta))) for i in range(n)])
#  expr1 = sum(logistic(X[i,:].T*beta*-Y[i]) for i in range(n))
#  expr1 = conv(1.0 / float(n), sum(expr1))
#  expr2 = square(norm(beta))
#  expr2 = conv(lamb, expr2)
#  expr_list = [expr1, expr2]
#  expr = sum(expr_list)
#  obj = Minimize(expr)
#
#  p = Problem(obj)
#  p.solve()
#  
#  beta_arr = np.array([b[0,0] for b in beta])
#  return beta, p.value
#
