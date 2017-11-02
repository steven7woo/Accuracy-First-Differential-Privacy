#!/usr/bin/python3

# This file collects all the "theory" bounds for various private ERM methods,
# i.e. bounds that say a given set of parameters has such an expected excess error.

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import traceback

usage_str = """
Usage: python3 theory.py n dim lamb norm

where:
  n     = number of data points
  dim   = dimension of each x-vector (where data points are (x,y)
  lamb  = lambda, regularization parameter
  norm  = maximum 2-norm of any hypothesis (0 to use 1/sqrt(lambda))

Plots alpha (excess empirical risk) on the horizontal axis
and epsilon (privacy parameter) on the vertical axis, for various
algorithms for least squares regression.

Uses the known theoretical guarantees for each algorithm to
compute the epsilon such that the algorithm is (epsilon, delta)-
differentially private and, with probability 1-gamma,
the average error on the dataset is no more than optimal plus
alpha.
"""

MAX_EPS = -1

# return the epsilon so that stochastic gradient descent
# is (epsilon, delta) private and has expected excess risk
# at most alpha
# For NON-STRONGLY-CONVEX
#
# Source: Theorem 2.4 of Bassily-Smith-Thakurta 2014
# https://arxiv.org/abs/1405.7085
# combined with Theorem 2 of Shamir-Zhang 2013
# http://proceedings.mlr.press/v28/shamir13.pdf
def sgd_get_epsilon_expected(alpha, n, dim, delta, diameter, lipschitz):
  paper_alpha = n*alpha  # paper uses non-normalized error
  a = 32.0 * dim * math.log(n/delta) * math.log(1.0/delta)
  b = 4.0 * (1.0 + math.log(n)) * diameter * lipschitz
  c = (paper_alpha / b)**2 - 1.0

  # an approximation from rounding the constants in the papers
  #approx = b * math.sqrt(a) / paper_alpha
  if c < 0:
    print("SGD Theory: n = " + str(n) + " but need n >= " + str(b/alpha) + " for meaningful bound.")
    return MAX_EPS
  eps = math.sqrt(a / c)
  return eps



# Ditto, but for lambda-strongly convex loss functions
def sgd_get_epsilon_expected_strongly(alpha, n, dim, delta, lipschitz, lamb):
  paper_alpha = n*alpha
  a = 32.0 * dim * math.log(n/delta) * math.log(1.0/delta)
  b = 17.0 * lipschitz**2 * (1.0 + 2.0*math.log(n))
  c = (paper_alpha * lamb / b) - 1.0
  if c < 0:
    print("SGD Theory: n = " + str(n) + " but need n >= " + str(b/(lamb*alpha)) + " for meaningful bound.")
    return MAX_EPS
  eps = math.sqrt(a / c)
  return eps



# return the epsilon so that the noisy-covariance method
# is epsilon private and  has expected excess risk at most alpha
def covar_get_epsilon(alpha, n, dim, max_norm):
  return 4.0*math.sqrt(2.0)*(2.0*math.sqrt(dim)*max_norm + dim*max_norm**2.0) / (n*alpha)


# return the epsilon so that the output perturbation method
# is epsilon private and  has expected excess risk at most alpha
def output_pert_linreg_get_epsilon(alpha, n, dim, lamb, max_norm):
  term1 = math.sqrt(1.0/n + lamb)
  term2 = math.sqrt(1.0 / (n*lamb*alpha))
  term3 = (max_norm+1.0)*dim
  return term1 * term2 * term3


# ditto but for output perturbation
def output_pert_logist_get_epsilon(alpha, n, dim, lamb):
  return 2.0 * math.sqrt(2.0) * dim / (n * lamb * alpha)





def main(n, dim, lamb, norm):
  two_norm = norm / math.sqrt(dim)
  delta = 1.0 / n
  alphas = np.linspace(0.001, 0.5, 1000)
  sgd = lambda a: sgd_get_epsilon_expected(a, n, dim, delta, 2.0*two_norm, 1.0)
  cov = lambda a: covar_get_epsilon(a, n, dim, two_norm)
  out = lambda a: output_pert_linreg_get_epsilon(a, n, dim, lamb, two_norm)

  methods = [sgd, cov, out]
  method_names = ["stochastic gradient descent", "covariance perturbation", "output perturbation"]
  colors = ["black", "red", "blue"]
  linewidth = 2.5

  plt.figure()
  for i,method in enumerate(methods):
    plt.plot(alphas, list(map(method,alphas)), color = colors[i], linewidth=linewidth)
  plt.legend(method_names)
  plt.show()


if __name__ == "__main__":
  try:
    n = int(sys.argv[1])
    dim = int(sys.argv[2])
    lamb = float(sys.argv[3])
    norm = float(sys.argv[4])
    if norm <= 0.0:
      norm = 1.0 / math.sqrt(lamb)

    main(n, dim, lamb, norm)

  except:
    print(usage_str)
    print("--------")
    print(traceback.format_exc())
    exit()
  


