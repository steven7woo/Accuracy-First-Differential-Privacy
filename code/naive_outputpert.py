# implements a naive output perturbation scheme
# that doubles epsilon and checks the error until good enough


import numpy as np
import math

# our file
import project


THRESHOLD_FRACTION = 0.5


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


def construct_beta_hats(opt_beta, opt_beta_sens, eps_list, max_norm):
  beta_hats = gen_list(opt_beta, opt_beta_sens, eps_list)
  for i in range(len(beta_hats)):
    beta_hats[i] = project.two_norm_project(beta_hats[i], max_norm)
  return beta_hats


# return threshold, epsilon so that
# with prob 1-gamma, max of num_steps independent Laplaces(sensitivity/eps)
# is at most alpha-threshold
def compute_test_epsilon(alpha, gamma, sens, num_steps):
  thresh = THRESHOLD_FRACTION * alpha
  # Laplace parameter
  b = (alpha - thresh) / math.log(num_steps / (2.0*gamma))
  eps = sens / b
  return thresh, eps


# compute_err_func is the loss function,
# see "get_excess_err_func" above
def run_naive(opt_beta, alpha, gamma, max_norm, min_eps, max_eps, sv_sens, opt_beta_sens, data, compute_err_func):
  max_step = int(math.log2(max_eps / min_eps) + 1.0)
  test_thresh, test_eps = compute_test_epsilon(alpha, gamma, sv_sens, max_step+1.0)
  eps_list = np.array([min_eps * 2.0**k for k in range(max_step+1)])

  beta_hats = construct_beta_hats(opt_beta, opt_beta_sens, eps_list, max_norm)
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


