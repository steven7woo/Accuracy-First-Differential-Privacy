import sys
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# our implementations
import run_ridge
import ridge
import theory
import covar
import output_pert
import naive_covar


#MIN_EPS = 0.00001
#MAX_EPS_CAP = 20.0
#MAX_NAIVE_EPS = 1000.0  # this one can be larger due to doubling


usage_str = """
Usage: python3 run_many.py datafilename lambda alpha gamma max_norm max_steps num_trials outputdir

Runs 'num_trials' separate trials, writing the output to the files
outputdir/trial_1.txt, outputdir/trial_2.txt, ....

--------
For other parameters: 
""" + run_ridge.usage_str




def main(X, Y, lamb, alpha, gamma, max_norm, max_steps, num_trials, outputdir):
  n = len(X)
  dim = len(X[0])
  if max_norm <= 0.0:
    max_norm = ridge.compute_max_norm(lamb)
  sv_sens = ridge.get_sv_sensitivity(max_norm, n)
  opt_beta_sens = ridge.compute_opt_sensitivity(n, dim, lamb, max_norm)
  compute_err_func = lambda X,Y,beta_hat: ridge.compute_err(X, Y, lamb, beta_hat)

  # Compute opt
  Sigma, R, opt_beta, opt_res = run_ridge.get_matrices_and_opt(X, Y, lamb)
  opt_err = opt_res[0]

  data = (X, Y, opt_err)
  min_eps = 1.0 / n
  max_covar_eps = 4.0 * theory.covar_get_epsilon(alpha, n, dim, max_norm)
  max_naive_eps = max_covar_eps
  max_output_eps = 4.0 * theory.output_pert_linreg_get_epsilon(alpha, n, dim, lamb, max_norm)

  # Create output folder and write value of alpha
  os.makedirs(outputdir)
  with open(outputdir + "/alpha.txt", "w") as f:
    f.write(str(alpha) + "\n")

  # Compute results of methods and save them
  for trial_ind in range(num_trials):
    covar_beta_hat, covar_res = covar.run_covar(Sigma, R, alpha, gamma, max_norm, max_steps, min_eps, max_covar_eps, sv_sens, data, compute_err_func)
    
    output_beta_hat, output_res = output_pert.run_output_pert(opt_beta, alpha, gamma, max_norm, max_steps, min_eps, max_output_eps, sv_sens, opt_beta_sens, data, compute_err_func)
    
    naive_beta_hat, naive_res = naive_covar.run_naive(Sigma, R, alpha, gamma, max_norm, min_eps, max_naive_eps, sv_sens, data, compute_err_func)

    with open(outputdir + "/trial_" + str(trial_ind+1) + ".txt", "w") as f:
      f.write(run_ridge.stringify(opt_res))
      f.write("\n")
      f.write(run_ridge.stringify(opt_beta))
      f.write("\n")
      for beta, res in [(covar_beta_hat, covar_res), (output_beta_hat, output_res), (naive_beta_hat, naive_res)]:
        success, excess_err, sv_eps, my_eps, index = res
        two_norm = np.linalg.norm(beta)
        mse = ridge.compute_err(X, Y, 0.0, beta)
        f.write(run_ridge.stringify(("1" if success else "0", excess_err, sv_eps, my_eps, index, two_norm, mse)))
        f.write("\n")
        f.write(run_ridge.stringify(beta))
        f.write("\n")





# when run as script, read parameters from input
# (other python scripts can call main(), above, directly)
if __name__ == "__main__":
  X, Y, lamb, alpha, gamma, max_norm, max_steps = run_ridge.parse_inputs(sys.argv)
  try:
    num_trials = int(sys.argv[7])
    outputdir = sys.argv[8]
  except:
    print(usage_str)
    exit(0)
    
    
  main(X, Y, lamb, alpha, gamma, max_norm, max_steps, num_trials, outputdir)

