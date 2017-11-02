import sys
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# our implementations
import ridge
import theory
import covar
import output_pert
import naive_covar
import sgd_ridge


#MIN_EPS = 0.00001
#MAX_EPS_CAP = 20.0
#MAX_NAIVE_EPS = 1000.0  # this one can be larger due to doubling


usage_str = """
Usage: python3 run_ridge.py filename lambda alpha gamma max_norm max_steps

Where:
  filename:  contains the matrix (X, Y); n rows each with d+1 space-separated real numbers
  lambda:    regularization parameter
  alpha:     accuracy, positive number
  gamma:     failure probability, in (0,1)
  max_norm:  maximum L2-norm of the hypothesis (if 0, use sqrt(1/lambda))
  max_steps: number of noise-reduction steps to take

The empirical risk is measured with the function
    0.5 * (1/n) * sum_i (y_i - beta*x_i)^2  +  0.5 * lambda ||beta||_2^2
where beta is the hypothesis.

Prints 9 lines.
Line 1 has four space-separated numbers
  * the error of the optimal hypothesis for ridge objective
  * the two-norm of the optimal hypothesis
  * the mean squared error of the optimal hypothesis
  * the minimum possible mean squared error for any hypothesis
Line 2 is the optimal hypothesis, space-separated
Line 3 has five space-separated numbers for the covariance method:
  * 1 if covariance method succeeded, 0 otherwise
  * excess error of covariance method's hypothesis
  * AboveThreshold epsilon used
  * empirical epsilon (total privacy is the sum of this and AT epsilon)
  * index at which AboveThreshold stopped
  * two-norm of hypothesis
  * mean squared error of hypothesis
Line 4 is the covariance method's hypothesis.
Lines 5 and 6 are the same as 3 and 4, but for output perturbation.
Lines 6 and 7 are the same as 3 and 4, but for a naive-covariance method.
Lines 8 and 9 are the same as 3 and 4, but for a naive-stochastic-gradient-descent method.
"""



def stringify(arr):
  return " ".join(list(map(str, arr)))



# ---------------------------------------------

# return Sigma, R, opt_beta, opt_res
# where opt_res = (opt's error, opt's two-norm, opt's unregularized err, best possible unreg. err)
def get_matrices_and_opt(X, Y, lamb):
  Sigma = ridge.compute_Sigma(X, lamb)
  R = ridge.compute_R(X, Y)
  opt_beta = ridge.exact_solve(Sigma, R)
  opt_err = ridge.compute_err(X, Y, lamb, opt_beta)

  opt_mse = ridge.compute_err(X, Y, 0.0, opt_beta)
  mse_opt_beta = ridge.exact_solve_data(X, Y, 0.0)
  mse_opt_mse = ridge.compute_err(X, Y, 0.0, mse_opt_beta)

  opt_res = (opt_err, np.linalg.norm(opt_beta), opt_mse, mse_opt_mse)
  return Sigma, R, opt_beta, opt_res


def main(X, Y, lamb, alpha, gamma, max_norm, max_steps):
  # Compute parameters etc
  n = len(X)
  dim = len(X[0])
  if max_norm <= 0.0:
    max_norm = ridge.compute_max_norm(lamb)
  sv_sens = ridge.get_sv_sensitivity(max_norm, n)
  opt_beta_sens = ridge.compute_opt_sensitivity(n, dim, lamb, max_norm)
  compute_err_func = lambda X,Y,beta_hat: ridge.compute_err(X, Y, lamb, beta_hat)

  # Compute opt
  Sigma, R, opt_beta, opt_res = get_matrices_and_opt(X, Y, lamb)
  opt_err = opt_res[0]

  data = (X, Y, opt_err)
  min_eps = 1.0 / n
  max_covar_eps = 4.0 * theory.covar_get_epsilon(alpha, n, dim, max_norm)
  max_naive_eps = max_covar_eps
  max_output_eps = 4.0 * theory.output_pert_linreg_get_epsilon(alpha, n, dim, lamb, max_norm)

  # Compute results of methods
  covar_beta_hat, covar_res = covar.run_covar(Sigma, R, alpha, gamma, max_norm, max_steps, min_eps, max_covar_eps, sv_sens, data, compute_err_func)

  output_beta_hat, output_res = output_pert.run_output_pert(opt_beta, alpha, gamma, max_norm, max_steps, min_eps, max_output_eps, sv_sens, opt_beta_sens, data, compute_err_func)

  naive_beta_hat, naive_res = naive_covar.run_naive(Sigma, R, alpha, gamma, max_norm, min_eps, max_naive_eps, sv_sens, data, compute_err_func)

#  sgd_beta_hat, sgd_res = sgd_ridge.run_naive_sgd(opt_beta, alpha, gamma, max_norm, min_eps, max_naive_eps, sv_sens, data, compute_err_func, lamb)

  print(stringify(opt_res))
  print(stringify(opt_beta))
  for beta, res in [(covar_beta_hat, covar_res), (output_beta_hat, output_res), (naive_beta_hat, naive_res)]: #, (sgd_beta_hat, sgd_res)]:
    success, excess_err, sv_eps, my_eps, index = res
    two_norm = np.linalg.norm(beta)
    unreg_err = ridge.compute_err(X, Y, 0.0, beta)
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

