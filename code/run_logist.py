import sys
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# our implementations
import logist
import theory
import output_pert
import naive_outputpert


#MIN_EPS = 0.00001
#MAX_EPS_CAP = 20.0
#MAX_NAIVE_EPS = 1000.0  # this one can be larger due to doubling


usage_str = """
Usage: python3 run_logist.py filename lambda alpha gamma max_norm max_steps

Where:
  filename:  contains the matrix (X, Y); n rows each with d+1 space-separated real numbers
  lambda:    regularization parameter
  alpha:     accuracy, positive number
  gamma:     failure probability, in (0,1)
  max_norm:  maximum L2-norm of the hypothesis (if 0, use theory bound)
  max_steps: number of noise-reduction steps to take

The empirical risk is measured with the function
    (1/n) * sum_i log(1.0 + exp(y_i * (beta*x_i)))  +  0.5 * lambda ||beta||_2^2
where beta is the hypothesis.

Prints 6 lines.
Line 1 has four space-separated numbers
  * the error of the optimal hypothesis for ridge objective
  * the two-norm of the optimal hypothesis
  * the mean squared error of the optimal hypothesis
  * the minimum possible mean squared error for any hypothesis
Line 2 is the optimal hypothesis, space-separated
Line 3 has five space-separated numbers for output perturbation:
  * 1 if covariance method succeeded, 0 otherwise
  * excess error of covariance method's hypothesis
  * AboveThreshold epsilon used
  * empirical epsilon (total privacy is the sum of this and AT epsilon)
  * index at which AboveThreshold stopped
  * two-norm of hypothesis
  * mean squared error of hypothesis
Line 4 is output perturbation's hypothesis.
Lines 5 and 6 are the same as 3 and 4, but for a naive algorithm.
"""



def stringify(arr):
  return " ".join(list(map(str, arr)))



# ---------------------------------------------

# return opt_beta, opt_res
# where opt_res = (opt's error, opt's two-norm, opt's unregularized err, best possible unreg. err)
def get_opt(X, Y, lamb):
  n = len(X)
  opt_beta, opt_err = logist.logistic_regression(X, Y, lamb)

  opt_unreg_err = logist.compute_err(X, Y, 0.0, opt_beta)
  unreg_opt, unreg_opt_err = logist.logistic_regression(X, Y, 1.0/(n*n))

  opt_res = (opt_err, np.linalg.norm(opt_beta), opt_unreg_err, unreg_opt_err)
  return opt_beta, opt_res



def main(X, Y, lamb, alpha, gamma, max_norm, max_steps):
  # Compute parameters etc
  n = len(X)
  dim = len(X[0])
  if max_norm <= 0.0:
    max_norm = logist.compute_max_norm(lamb)
  sv_sens = logist.get_sv_sensitivity(max_norm, n)
  opt_beta_sens = logist.compute_opt_sensitivity(n, dim, lamb)
  compute_err_func = lambda X,Y,beta_hat: logist.compute_err(X, Y, lamb, beta_hat)

  # Compute opt
  opt_beta, opt_res = get_opt(X, Y, lamb)
  opt_err = opt_res[0]

  data = (X, Y, opt_err)
  min_eps = 1.0 / n   # theory epsilon scales with 1/n
  max_output_eps = 4.0 * theory.output_pert_logist_get_epsilon(alpha, n, dim, lamb)
  max_naive_eps = max_output_eps

  # Compute results of methods
  output_beta_hat, output_res = output_pert.run_output_pert(opt_beta, alpha, gamma, max_norm, max_steps, min_eps, max_output_eps, sv_sens, opt_beta_sens, data, compute_err_func)

  naive_beta_hat, naive_res = naive_outputpert.run_naive(opt_beta, alpha, gamma, max_norm, min_eps, max_output_eps, sv_sens, opt_beta_sens, data, compute_err_func)

  print(stringify(opt_res))
  print(stringify(opt_beta))
  for beta, res in [(output_beta_hat, output_res), (naive_beta_hat, naive_res)]:
    success, excess_err, sv_eps, my_eps, index = res
    two_norm = np.linalg.norm(beta)
    unreg_err = logist.compute_err(X, Y, 0.0, beta)
    print("1" if success else "0", excess_err, sv_eps, my_eps, index, two_norm, unreg_err)
    print(stringify(beta))



# returns:
#   X = list of data point features (vectors of dimension 'dim')
#   Y = list of data point labels (real numbers)
#   lamb = regularization parameter
#   alpha = target excess empirical risk
#   gamma = maximum failure probability
#   max_norm = maximum norm of hypothesis
#   max_steps = maximum number of AboveThresh iterations (size of epsilon grid)
#
def parse_inputs(args):
  try:
    filename = args[1]
    lamb = float(args[2])
    alpha = float(args[3])
    gamma = float(args[4])
    max_norm = float(args[5])
    max_steps = int(args[6])
    X, Y = [], []
    with open(filename) as f:
      for line in f:
        nums = [float(x) for x in line.split()]
        X.append(nums[:-1])
        Y.append(nums[-1])
    X = np.array(X)
    Y = np.array(Y)
  except:
    print(usage_str)
    exit(0)

  return X, Y, lamb, alpha, gamma, max_norm, max_steps



# when run as script, read parameters from input
# (other python scripts can call main(), above, directly)
if __name__ == "__main__":
  X, Y, lamb, alpha, gamma, max_norm, max_steps = parse_inputs(sys.argv)
  main(X, Y, lamb, alpha, gamma, max_norm, max_steps)

