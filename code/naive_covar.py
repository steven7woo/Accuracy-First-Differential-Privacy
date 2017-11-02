# implements a naive scheme using the covariance method
# that doubles epsilon and checks the error until good enough

import numpy as np
import math


# our files
import project
import ridge


# Only accept if noisy error is below THRESHOLD_FRAC * alpha
# (and bound the chance that any noise variable exceeds (1 - THRESHOLD_FRAC)*alpha
THRESHOLD_FRAC = 0.5


# get a function f(data) -> excess error of beta_hat on data
# where data is a 3-tuple: X,Y,opt_err
# and compute_err_func(X, Y, beta_hat) -> error
def get_excess_err_func(beta_hat, compute_err_func):
  def f(data):
    X, Y, opt_err = data
    err = compute_err_func(X, Y, beta_hat)
    return err - opt_err
  return f


# Input:
#   matrix M
#   sensitivity of M (how much each entry changes from D to D')
#   list of epsilons
# Output: a list of Mhat matrices approximating M
# where releasing ONLY the t-th matrix is eps_t private
# using independent noise.
def gen_list(M, sens, eps_list):
  return [M + np.random.laplace(scale=sens/eps, size=M.shape) for eps in eps_list]


def construct_beta_hats(Sigma, R, eps_list, max_norm):
  # to be eps-private, we need each Sigma and R to be (eps/2)-private
  halved_eps_list = [eps/2.0 for eps in eps_list]
  Sigma_hats = gen_list(Sigma, 2.0, halved_eps_list)
  R_hats = gen_list(R, 2.0, halved_eps_list)
  beta_hats = np.array([ridge.exact_solve_and_project(S_hat, R_hat, max_norm) for S_hat,R_hat in zip(Sigma_hats, R_hats)])
  return beta_hats


# return threshold, epsilon so that
# with prob 1-gamma, max of num_steps independent Laplaces(sensitivity/eps)
# is at most alpha-threshold
def compute_test_epsilon(alpha, gamma, sens, num_steps):
  thresh = alpha * THRESHOLD_FRAC
  # Laplace parameter
  b = (alpha - thresh) / math.log(num_steps / (2.0*gamma))
  eps = sens / b
  return thresh, eps


# compute_err_func is the loss function,
# see "get_excess_err_func" above
def run_naive(Sigma, R, alpha, gamma, max_norm, min_eps, max_eps, sv_sens, data, compute_err_func):
  max_step = int(math.log2(max_eps / min_eps) + 1.0)
  test_thresh, test_eps = compute_test_epsilon(alpha, gamma, sv_sens, max_step+1.0)
  eps_list = np.array([min_eps * 2.0**k for k in range(max_step+1)])

  beta_hats = construct_beta_hats(Sigma, R, eps_list, max_norm)
  queries = np.array([get_excess_err_func(b, compute_err_func) for b in beta_hats])
  result = -1
  for t,eps in enumerate(eps_list):
    if queries[t](data) + np.random.laplace(scale=sv_sens/test_eps) <= test_thresh:
      result = t
      break
  if result == -1:  # failure
    return beta_hats[-1], (False, queries[-1](data), -1, -1, -1)
  else:
    return beta_hats[result], (True, queries[result](data), test_eps*(result+1), sum(eps_list[:result+1]), result+1)




