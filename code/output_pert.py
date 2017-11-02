# output perturbation method
# add noise to the optimal solution vector

import numpy as np
import math

# our files
import sparsevec
import noise_reduc
import project


# get a function f(data) -> excess error of beta_hat on data
# where data is a 3-tuple: X,Y,opt_err
def get_excess_err_func(beta_hat, compute_err_func):
  def f(data):
    X, Y, opt_err = data
    err = compute_err_func(X, Y, beta_hat)
    return err - opt_err
  return f


# compute the list of beta_hats for noise reduction
# n = # data points
# dim = dimension of beta
# lamb = lambda parameter for ridge regression
def construct_beta_hats(opt_beta, sensitivity, eps_list, max_norm):
  beta_hats = noise_reduc.gen_list(opt_beta, sensitivity, eps_list)
  for i in range(len(beta_hats)):
    beta_hats[i] = project.two_norm_project(beta_hats[i], max_norm)
  return beta_hats


# Return: beta_hat, algo_res
# where algo_res is a tuple containing:
#    success (True or False)
#    excess error
#    sparsevec epsilon
#    noise-reduc epsilon
#    stopping index (number of AboveThresh iterations)
def run_output_pert(opt_beta, alpha, gamma, max_norm, max_steps, min_eps, max_eps, sv_sens, opt_beta_sens, data, compute_err_func):
  sv_thresh, sv_eps = sparsevec.compute_epsilon(alpha, gamma, sv_sens, max_steps, True)
  const = (max_eps / min_eps)**(1.0 / max_steps)
  eps_list = np.array([min_eps * const**k for k in range(max_steps)])
#  eps_list = np.linspace(min_eps, max_eps, max_steps)
  beta_hats = construct_beta_hats(opt_beta, opt_beta_sens, eps_list, max_norm)

  queries = np.array([get_excess_err_func(b, compute_err_func) for b in beta_hats])
  stop_ind = sparsevec.below_threshold(data, queries, sv_thresh, sv_eps, sv_sens)
  if stop_ind == -1:  # failure
    return beta_hats[-1], (False, queries[-1](data), -1, -1, -1)
  else:
    return beta_hats[stop_ind], (True, queries[stop_ind](data), sv_eps, eps_list[stop_ind], stop_ind)


