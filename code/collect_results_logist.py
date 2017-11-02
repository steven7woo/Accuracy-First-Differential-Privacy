#!/usr/bin/python3

# Collect all simulation results from a directory where
# they were saved and plot them

import os, sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 

matplotlib.rc('xtick', labelsize=11) 
matplotlib.rc('ytick', labelsize=11)
matplotlib.rc('legend', fontsize=14)
matplotlib.rc('axes', titlesize=14)
matplotlib.rc('axes', labelsize=14)

# our implementations
import theory
import logist


usage_str = """
Usage: python3 collect_results_ridge.py sim-results

where the argument "sim-results" is the directory where
the output of simulations is saved.
"""


opt_color = "red"
noisered_color = "blue"
noisered2_color = "purple"
naive_color = "darkgreen"
sv_eps_color = "orange"
nr_eps_color = "brown"
theory1_color = "pink"
theory2_color = "green"
theory3_color = "orange"
theory4_color = "brown"
opt_linewidth = 1.0
linewidth = 2.0

opt_name = "Opt"
noisered_name = "NoiseReduction"
output_name = "OutputPert"
naive_name = "Doubling"

avg_loss_title_name = "Average ex post privacy loss $\epsilon$"
loss_title_name = "Ex post privacy loss $\epsilon$"
avg_risk_title_name = "Average ex post privacy risk $\exp[\epsilon]$"
risk_title_name = "Ex post privacy loss $\exp[\epsilon]$"

xaxis_name = "Input $\\alpha$ (excess error guarantee)"
eps_yaxis_name = "ex-post privacy loss $\epsilon$"
risk_yaxis_name = "ex-post privacy risk factor $\exp[\epsilon]$"


# return e^x unless overflow, then return M
def myexp(x, M=-1.0):
  try:
    return math.exp(x)
  except OverflowError:
    return M

# filter the list by only including successes
def filter_succ(s_trials, successes):
  return [s for s,succ in zip(s_trials,successes) if succ==1]

# get average of each component of statlist that was a success
# use placeholder -1 if no successes
# operation, if not None, transforms each data point before averaging
def get_avg(statlist, succlist, op=None):
  if op is None:
    op = lambda x: x
  filts = [[op(x) for x in filter_succ(a,b)] for a,b in zip(statlist, succlist)]
  return [np.average(filt) if len(filt) > 0 else -1 for filt in filts]

# get xth percentile of each component that was a success
def get_percentile(frac, statlist, succlist):
  filts = [sorted(filter_succ(a,b)) for a,b in zip(statlist, succlist)]
  return [s[int(frac*len(s))] if len(s) > 0 else -1 for s in filts]


def myplot(xs, ys, **kwargs):
  non_fail_prs = np.array([(x,y) for x,y in zip(xs,ys) if y >= 0])
  if len(non_fail_prs) > 0:
    plt.plot(non_fail_prs[:,0], non_fail_prs[:,1], **kwargs)





# ---------------------------------------------


# get variables n, d (dimension), lamb, gamma, max_norm, max_steps
if len(sys.argv) <= 1:
  print(usage_str)
  exit(0)

dirname = sys.argv[1] + "/"
exec(open(dirname + "about.py").read(), globals())
if max_norm <= 0.0:
  max_norm = logist.compute_max_norm(lamb)

# The following lists will have one entry per alpha
# alphas = [alpha_1, alpha_2, ...]
# others: entry i,j is a list of numtrials results for alpha_i, result #j
alphas       = []
opt_datas    = []
opt_betas    = []
output_datas = []
output_betas = []
naive_datas  = []
naive_betas  = []

# index into each entry of the above data arrays
OPT_ERR = 0
OPT_NORM = 1
OPT_MSE = 2      # mean squared error
MSE_OPT_MSE = 3  # optimal possible MSE
SUCC = 0         # success
EXCESS_ERR = 1
SV_EPS = 2       # epsilon due to sparse vector
NR_EPS = 3       # epsilon due to noise reduction
IND = 4          # index (step) of alg we halted on
NORM = 5
MSE = 6


# each list here stores a line in an output file
data_arrays = [opt_datas, opt_betas, output_datas, output_betas, naive_datas, naive_betas]

param_ind = 1
while True:
  mydir = dirname + "param-" + str(param_ind) + "/"
  if not os.path.isdir(mydir):
    break

  with open(mydir + "alpha.txt") as f:
    alphas.append(float(f.readline().strip()))
  for da in data_arrays:
    da.append([])
  trial_ind = 1
  while True:
    fname = mydir + "trial_" + str(trial_ind) + ".txt"
    if not os.path.exists(fname):
      break
    with open(fname) as f:
      for da in data_arrays:
        words = f.readline().strip().split()
        da[param_ind-1].append(list(map(float, words)))
    trial_ind += 1
  param_ind += 1

opt_datas    = np.array(opt_datas)
output_datas = np.array(output_datas)
naive_datas  = np.array(naive_datas)

output_succs = output_datas[:,:,SUCC]
naive_succs = naive_datas[:,:,SUCC]


print("Fraction of successes:")
for i in range(len(alphas)):
  print(output_name + " = " + str(sum(output_succs[i]) / float(len(output_succs[i]))) + ", " + naive_name + " = " + str(sum(naive_succs[i]) / float(len(naive_succs[i]))))

# ERROR plot
opt_err = np.average([np.average(elist) for elist in opt_datas[:,:,OPT_ERR]])
avg_output_excess_errs = get_avg(output_datas[:,:,EXCESS_ERR], output_succs)
avg_naive_excess_errs = get_avg(naive_datas[:,:,EXCESS_ERR], naive_succs)

plt.figure()
myplot(alphas, [opt_err]*len(alphas), color=opt_color, linewidth=opt_linewidth, label=opt_name)
myplot(alphas, [opt_err + a for a in avg_output_excess_errs], color=noisered_color, linewidth=linewidth, label=noisered_name)
myplot(alphas, [opt_err + a for a in avg_naive_excess_errs], color=naive_color, linewidth=linewidth, label=naive_name)
myplot(alphas, [opt_err + a for a in alphas], color='black', linewidth=1.0, linestyle="--", label=opt_name + " + $\\alpha$")

max_value = opt_err+alphas[-1]
plt.ylim([-0.05*max_value, 1.1*max_value])
plt.legend()
plt.title("Average regularized mean squared error")
plt.xlabel(xaxis_name)
plt.ylabel("error")


# ERROR plot - with 90th percentile
output_90th_excess_errs = get_percentile(0.9, output_datas[:,:,EXCESS_ERR], output_succs)

plt.figure()
myplot(alphas, [opt_err]*len(alphas), color=opt_color, linewidth=opt_linewidth, label=opt_name)
myplot(alphas, [opt_err + a for a in avg_output_excess_errs], color=noisered_color, linewidth=linewidth, label=noisered_name + " - average")
myplot(alphas, [opt_err + a for a in output_90th_excess_errs], color=noisered_color, linewidth=opt_linewidth, linestyle="--", label=noisered_name + " - 90th percentile")
myplot(alphas, [opt_err + a for a in alphas], color='black', linewidth=1.0, linestyle="--", label=opt_name + " + $\\alpha$")

max_value = opt_err+alphas[-1]
plt.ylim([-0.05*max_value, 1.1*max_value])
plt.legend()
plt.title("Regularized mean squared error")
plt.xlabel(xaxis_name)
plt.ylabel("error")



# MEAN SQUARED ERROR plot
opt_mse = np.average([np.average(elist) for elist in opt_datas[:,:,OPT_MSE]])
mse_opt_mse = np.average([np.average(elist) for elist in opt_datas[:,:,MSE_OPT_MSE]])
avg_output_mses = get_avg(output_datas[:,:,MSE], output_succs)
avg_naive_mses = get_avg(naive_datas[:,:,MSE], naive_succs)

plt.figure()
myplot(alphas, [opt_mse]*len(alphas), color=opt_color, linewidth=opt_linewidth, label=opt_name)
myplot(alphas, avg_output_mses, color=noisered_color, linewidth=linewidth, label=noisered_name)
myplot(alphas, avg_naive_mses, color=naive_color, linewidth=linewidth, label=naive_name)
myplot(alphas, [mse_opt_mse]*len(alphas), color="black", linewidth=opt_linewidth, linestyle="--", label="minimum possible MSE")

max_value = opt_err+alphas[-1]
plt.ylim([-0.05*max_value, 1.1*max_value])
plt.legend()
plt.title("Average mean squared error")
plt.xlabel(xaxis_name)
plt.ylabel("MSE")



# PRIVACY vs theory plot
avg_output_sv_epss = get_avg(output_datas[:,:,SV_EPS], output_succs, lambda x: 2.0*x)  # FACTOR 2 BECAUSE ALGS WERE UNDER-ESTIMATING
avg_output_nr_epss = get_avg(output_datas[:,:,NR_EPS], output_succs)
avg_naive_sv_epss = get_avg(naive_datas[:,:,SV_EPS], naive_succs, lambda x: 2.0*x)
avg_naive_nr_epss = get_avg(naive_datas[:,:,NR_EPS], naive_succs)

sgd_theory_strongly = [theory.sgd_get_epsilon_expected_strongly(alpha, n, dim, 1.0/n, max_norm + 1.0 + max_norm*lamb, lamb) for alpha in alphas]
sgd_theory_nonstrongly = [theory.sgd_get_epsilon_expected(alpha, n, dim, 1.0/n, 2.0*max_norm, max_norm + 1.0 + max_norm*lamb) for alpha in alphas]
output_theory = [theory.output_pert_linreg_get_epsilon(alpha, n, dim, lamb, max_norm) for alpha in alphas]

plt.figure()
myplot(alphas, sgd_theory_strongly, color=theory1_color, label="SGD theory (strongly convex)")
myplot(alphas, sgd_theory_nonstrongly, color=theory2_color, label="SGD theory (non-strongly convex)")
myplot(alphas, output_theory, color=theory4_color, label=output_name + " theory")
myplot(alphas, np.add(avg_output_sv_epss, avg_output_nr_epss), color=noisered_color, linewidth=linewidth, label=noisered_name)
#myplot(alphas, np.add(avg_naive_sv_epss, avg_naive_nr_epss), color=naive_color, linewidth=linewidth, label=naive_name)
plt.legend()
plt.title("Comparison to theory approach")
plt.xlabel(xaxis_name)
plt.ylabel(eps_yaxis_name)


# PRIVACY vs theory with exp(eps)
avg_output_sv_exp_epss = get_avg(output_datas[:,:,SV_EPS], output_succs,myexp)
avg_output_nr_exp_epss = get_avg(output_datas[:,:,NR_EPS], output_succs,myexp)
avg_naive_sv_exp_epss  = get_avg(naive_datas[:,:,SV_EPS], naive_succs,myexp)
avg_naive_nr_exp_epss  = get_avg(naive_datas[:,:,NR_EPS], naive_succs,myexp)
avg_output_exp_epss = get_avg(np.add(output_datas[:,:,SV_EPS],output_datas[:,:,NR_EPS]), output_succs,myexp)
avg_naive_exp_epss  = get_avg(np.add(naive_datas[:,:,NR_EPS],naive_datas[:,:,SV_EPS]), naive_succs,myexp)

# keep "failure" values of -1, otherwise exponentiate
MAX_NUM = 10000000
def exp_list(arr):
  return [myexp(a, MAX_NUM) if a >= 0.0 else a for a in arr]



plt.figure()
myplot(alphas, exp_list(sgd_theory_strongly), color=theory1_color, label="SGD theory (strongly convex)")
myplot(alphas, exp_list(sgd_theory_nonstrongly), color=theory2_color, label="SGD theory (non-strongly convex)")
myplot(alphas, exp_list(output_theory), color=theory4_color, label=output_name + " theory")
myplot(alphas, avg_output_exp_epss, color=noisered_color, linewidth=linewidth, label=noisered_name)
#myplot(alphas, avg_naive_exp_epss, color=naive_color, linewidth=linewidth, label=naive_name)
plt.legend()
plt.title("Comparison to theory approach")
plt.xlabel(xaxis_name)
plt.ylabel(risk_yaxis_name)
plt.ylim([-1,200])


# PRIVACY plot 2 - no theory
plt.figure()
myplot(alphas, np.add(avg_naive_sv_epss, avg_naive_nr_epss), color=naive_color, linewidth=linewidth, label=naive_name)
myplot(alphas, np.add(avg_output_sv_epss, avg_output_nr_epss), color=noisered_color, linewidth=linewidth, label=noisered_name)
plt.legend()
plt.title("Comparison to " + naive_name)
plt.xlabel(xaxis_name)
plt.ylabel(eps_yaxis_name)


# PRIVACY plot 3 - no theory, exp scales

plt.figure()
myplot(alphas, avg_naive_exp_epss, color=naive_color, linewidth=linewidth, label=naive_name)
myplot(alphas, avg_output_exp_epss, color=noisered_color, linewidth=linewidth, label=noisered_name)
plt.legend()
plt.title("Comparison to " + naive_name)
plt.xlabel(xaxis_name)
plt.ylabel(risk_yaxis_name)
plt.ylim([-1,100])


# PRIVACY plot 4 - breakdown of epsilon
plt.figure()
myplot(alphas, np.add(avg_output_sv_epss, avg_output_nr_epss), color=noisered_color, linewidth=1.0, label=noisered_name)
myplot(alphas, avg_output_sv_epss, color=sv_eps_color, linewidth=1.0, label="$\epsilon$ due to AboveThreshold")
myplot(alphas, avg_output_nr_epss, color=nr_eps_color, linewidth=1.0, label="$\epsilon$ due to NoiseReduction")
plt.legend()
plt.title("Breakdown of privacy loss")
plt.xlabel(xaxis_name)
plt.ylabel(eps_yaxis_name)


# PRIVACY plot 5 - breakdown of epsilon, exp y-axis
plt.figure()
myplot(alphas, avg_output_exp_epss, color=noisered_color, linewidth=1.0, label=noisered_name)
myplot(alphas, avg_output_sv_exp_epss, color=sv_eps_color, linewidth=1.0, label="$\exp[\epsilon]$ due to AboveThreshold")
myplot(alphas, avg_output_nr_exp_epss, color=nr_eps_color, linewidth=1.0, label="$\exp[\epsilon]$ due to NoiseReduction")
plt.legend()
plt.title("Breakdown of privacy loss")
plt.xlabel(xaxis_name)
plt.ylabel(risk_yaxis_name)
plt.ylim([-1,100])




# NUMBER OF STEPS
avg_output_inds = get_avg(output_datas[:,:,IND], output_succs)
avg_naive_inds = get_avg(naive_datas[:,:,IND], naive_succs)

plt.figure()
myplot(alphas, avg_output_inds, color=noisered_color, linewidth=linewidth, label=noisered_name)
myplot(alphas, avg_naive_inds, color=naive_color, linewidth=linewidth, label=naive_name)
plt.legend()
plt.title("Number of iterations of algorithm")
plt.xlabel(xaxis_name)
plt.ylabel("iterations")


# NORM OF HYPOTHESES
avg_output_norms = get_avg(output_betas, output_succs, np.linalg.norm)
avg_naive_norms = get_avg(naive_betas, naive_succs, np.linalg.norm)

plt.figure()
myplot(alphas, [max_norm for a in alphas], linewidth=opt_linewidth, linestyle="--", label="Upper bound")
myplot(alphas, [np.linalg.norm(opt_betas[0][0]) for a in alphas], linewidth=opt_linewidth,label=opt_name)
myplot(alphas, avg_output_norms, color=noisered_color, linewidth=linewidth, label=noisered_name)
myplot(alphas, avg_naive_norms, color=naive_color, linewidth=linewidth, label=naive_name)
plt.legend()
plt.title("Norms of Final Hypotheses")
plt.xlabel(xaxis_name)
plt.ylabel("iterations")





plt.show()

