# The "Covariance" method for privately solving ridge regression,
# combined with the noise reduction technique.

import numpy as np
import math

# our files
import sparsevec
import noise_reduc
import ridge
import theory



# get a function f(data) -> excess error of beta_hat on data
# where data is a 3-tuple: X,Y,opt_err
# and compute_err_func(X, Y, beta_hat) -> error
def get_excess_err_func(beta_hat, compute_err_func):
  def f(data):
    X, Y, opt_err = data
    err = compute_err_func(X, Y, beta_hat)
    return err - opt_err
  return f


# get an upper bound on the epsilon we need to achieve alpha
def get_max_epsilon(n, lamb, alpha, gamma, dim, max_norm):
  eps_for_expected = theory.covar_get_epsilon(alpha, n, dim, max_norm)
  return 2.0 * eps_for_expected


# compute the list of beta_hats for noise reduction
def construct_beta_hats(Sigma, R, eps_list, max_norm):
  # to be eps-private, we need each Sigma and R to be (eps/2)-private
  halved_eps_list = [eps/2.0 for eps in eps_list]
  sigma_sensitivity = 2.0
  Sigma_hats = noise_reduc.gen_list(Sigma, sigma_sensitivity, halved_eps_list)
  r_sensitivity = 2.0
  R_hats = noise_reduc.gen_list(R, r_sensitivity, halved_eps_list)
  beta_hats = np.array([ridge.exact_solve_and_project(S_hat, R_hat, max_norm) for S_hat,R_hat in zip(Sigma_hats, R_hats)])
  return beta_hats


# Return: beta_hat, algo_res
# where algo_res is a tuple containing:
#    success (True or False)
#    excess error
#    sparsevec epsilon
#    noise-reduc epsilon
#    stopping index (number of AboveThresh iterations)
def run_covar(Sigma, R, alpha, gamma, max_norm, max_steps, min_eps, max_eps, sv_sens, data, compute_err_func):
  sv_thresh, sv_eps = sparsevec.compute_epsilon(alpha, gamma, sv_sens, max_steps, True)
  const = (max_eps / min_eps)**(1.0 / max_steps)
  eps_list = np.array([min_eps * const**k for k in range(max_steps)])
#  eps_list = np.linspace(min_eps, max_eps, max_steps)
  beta_hats = construct_beta_hats(Sigma, R, eps_list, max_norm)

  queries = np.array([get_excess_err_func(b, compute_err_func) for b in beta_hats])
  stop_ind = sparsevec.below_threshold(data, queries, sv_thresh, sv_eps, sv_sens)
  if stop_ind == -1:  # failure
    return beta_hats[-1], (False, queries[-1](data), -1, -1, -1)
  else:
    return beta_hats[stop_ind], (True, queries[stop_ind](data), sv_eps, eps_list[stop_ind], stop_ind)


