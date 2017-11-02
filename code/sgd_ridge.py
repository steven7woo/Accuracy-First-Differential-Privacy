#!/usr/bin/python3

import math
import random
import numpy as np

import project

# Only accept if noisy error is below THRESHOLD_FRAC * alpha
# (and bound the chance that any noise variable exceeds (1 - THRESHOLD_FRAC)*alpha
THRESHOLD_FRAC = 0.5



def compute_variance(n, epsilon, delta, lipschitz):
  num = 32.0 * lipschitz**2.0 * n**2.0 * math.log(n/delta) * math.log(1/delta)
  den = epsilon**2.0
  return num / den




# data is a 3-tuple: X,Y,opt_err
def get_excess_err(beta_hat, compute_err_func, data):
  X, Y, opt_err = data
  err = compute_err_func(X, Y, beta_hat)
  return err - opt_err


#def loss_subgradient(beta, x, y, n, lamb):
#  c = np.dot(beta, x) - y
#  return np.array([c*x[i] + (lamb/n)*beta[i] for i in range(len(beta))])


def sgd_ridge(data, epsilon, delta, lamb, max_norm):
  X, Y, opt_err = data
  n = len(X)
  lipschitz = max_norm + 1.0 + max_norm*lamb
  strong_convexity = lamb
  variance = compute_variance(n, epsilon, delta, lipschitz)
  sigma = math.sqrt(variance)
  beta = np.zeros(len(X[0]))
  indices = range(n)
  for t in range(1, n**2):
    data_ind = random.choice(indices)
    x, y = X[data_ind], Y[data_ind]
    noise = sigma * np.random.normal(size=(len(beta)))
    eta = 1.0 / (t * strong_convexity)
    # subtract eta * (subgradient + noise)
    c1 = np.dot(beta, x) - y
    c2 = lamb / n
    for i in range(len(beta)):
      beta[i] -= eta * (c1*x[i] + c2*beta[i] + noise[i])
    # project back into max_norm
    curr_norm = np.linalg.norm(beta)
    if curr_norm > max_norm:
      ratio = max_norm / curr_norm
      for i in range(len(beta)):
        beta[i] = ratio*beta[i]
  return beta


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
def run_naive_sgd(opt_beta, alpha, gamma, max_norm, min_eps, max_eps, sv_sens, data, compute_err_func, lamb):
  max_step = int(math.log2(max_eps / min_eps) + 1.0)
  test_thresh, test_eps = compute_test_epsilon(alpha, gamma, sv_sens, max_step+1.0)
  eps_list = np.array([min_eps * 2.0**k for k in range(max_step+1)])

  beta_hat = opt_beta
  excess_err = 0.0
  result = -1
  per_step_delta = 1.0 / (len(data[0]) * len(eps_list))
  for t,eps in enumerate(eps_list):
    beta_hat = sgd_ridge(data, eps, per_step_delta, lamb, max_norm)
    excess_err = get_excess_err(beta_hat, compute_err_func, data)
    if excess_err + np.random.laplace(scale=sv_sens/test_eps) <= test_thresh:
      result = t
      break
  if result == -1:  # failure
    return opt_beta, (False, excess_err, -1, -1, len(eps_list))
  else:
    return beta_hat, (True, excess_err, test_eps*(result+1), sum(eps_list[:result+1]), result+1)







