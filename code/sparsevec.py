import math
import random
import numpy as np

# The Sparse Vector algorithm as in Dwork and Roth 2014.
# NOTE! This code is used for academic simulations.
# It is not actually guaranteed to be private because it uses python's floating-point
# arithmetic and random numbers, whereas the theorems are proved for true real numbers
# and true randomness.

# Parameters
# Set the sparse vector threshold as this fraction of alpha
THRESHOLD_FRACTION = 0.5


# --Input--
#   data: a database
#   f_list: a list of functions of the form f(data) = real_number
#   sv_thresh: a real number
#   epsilon: guarantee epsilon-differential privacy
#   sensitivity: maximimum sensitivity of any query
# --Output-- 
#   index i of the first function in f_list that is approximately lower than sv_thresh
#   (return -1 if none)
def below_threshold(data, f_list, sv_thresh, epsilon, sensitivity):
  noisy_threshold = sv_thresh + np.random.laplace(scale=2.0*sensitivity / epsilon)
  for i, f in enumerate(f_list):
    if f(data) + np.random.laplace(scale=4.0*sensitivity / epsilon) <= noisy_threshold:
      return i
  return -1


# --Input--
#   alpha: the "real" threshold, a positive number
#   gamma: probability of failure
#   sensitivity: maximum sensitivity of any query
#   num_steps: the maximum number of time steps
#   auto_compute_threshold: see below
#
# --Output--
#   (sv_thresh, epsilon) to use in below_threshold
#   so that, with prob 1-gamma, below_threshold's answer is below alpha
# Here failure means that below_threshold returns no answer, or
# an answer whose accuracy is not as good as alpha.
#
def compute_epsilon(alpha, gamma, sensitivity, num_steps, auto_compute_threshold=True):
  if auto_compute_threshold:
    # use sv_thresh calculated to optimize epsilon
    b = alpha / (4.0 * math.log(1.5**0.75 * math.sqrt(num_steps) / gamma) )
    sv_thresh = 0.25*alpha + 0.75*b*math.log(1.5) - 0.5*b*math.log(num_steps)
  else:
    # use parameter for sv_thresh, then calculated value of b
    sv_thresh = alpha * THRESHOLD_FRACTION
    b = (alpha-sv_thresh) / (3.0*math.log( (0.5 + num_steps**(2.0/3))/gamma ))

  # need to add noise with laplace parameter b = 2*sensitivity/epsilon
  # for threshold and parameter 2b = 4*sensitivity/epsilon for queries
  eps = 2.0 * sensitivity / b
  return sv_thresh, eps




# auto_compute_threshold
# If True, set the sparse-vector threshold as the value that maximizes
# the privacy obtained from the sparse vector method
# (note this will probably be a very low threshold, so it will usually
# end up with a much higher accuracy






# CALCULATIONS
#
# Given alpha, gamma, and num_steps, either:
#  - use the fixed sv_thresh = alpha*THRESHOLD_FRACTION, or
#  - calculate the optimal sv_thresh myopically optimizing SV's privacy (even though
#    it may set the threshold low)
#
# and calculate the Laplace noise parameter b.
# Goal: with probability at least 1-gamma,
# we don't "fail", meaning the noisy threshold goes below 0 or the noisy_threshold plus
# some query noise goes above alpha.
#
# Here the noisy threshold is sv_thresh + Lap(b)
# and each query noise is Lap(2b).
#
# So always, epsilon = 2 * sensitivity / b.
#
# -------------
# 
# 1. Fixed THRESHOLD_FRACTION
# 
# Determining b as a function of sv_thresh, num_steps, and gamma.
#
# Let b = 2*sensitivity/epsilon. T is the total number of time steps and gamma is the failure probability.
# 
#     gamma <= Pr[ noisy_thresh <= 0 ]  + Pr[ noisy_thresh + max(query_noise) >= alpha ].
#     gamma = 0.5*exp[ -sv_thresh/b ]  +  TERM.
# 
# Now, TERM <= Pr[ thresh_noise > z*(alpha-sv_thresh) ] + Pr[ exists t with query_noise(t) > (1-z)*(alpha-sv_thresh) ]  
#           <= 0.5*exp[ -z*(alpha - sv_thresh)] + T*Pr[ query_noise(1) > (1-z)*(alpha - sv_thresh) ]
#            = 0.5*[ exp[-z*(a-s)/b]  +  T*exp[-(1-z)(a-s)/(2b)] ].
# 
# This holds for all z in (0,1). Using calculus, we find the optimal z = 1/3 - (2b/(3(a-s)))*ln(T). (at the optimal z, both terms are equal to 0.5*exp[-(a-s)/(3b) + (2/3)ln(T)].)
# 
#     gamma <= 0.5*exp[ -s/b ] + T^(2/3)exp[-(a-s)/(3b)]
#            = (0.5*exp[(a-4s)/3b] + T^(2/3)) * exp[-(a-s)/(3b)].
# Now for sv_thresh >= a/4, we get
#     gamma <= (0.5 + T^(2/3)) * exp[-(a-s)/(3b)]
# so
#     b <= (a-s) / 3ln( (0.5 + T^(2/3))/gamma ).
# 
# ----------
# 
# USE_OPT_THRESHOLD
#
# We can go further and compute the optimal sv_threshold s, if we didn't care about the effect setting it low would have on the noise reduction technique.
# 
#     gamma <= 0.5*exp[ -s/b ] + T^(2/3)exp[-(a-s)/(3b)].
# 
# Using calculus to find the optimal s in [0,a], we get s = (a/4) + (3/4)*b*ln(1.5) - (1/2)*b*ln(T)).
#     gamma <=    exp[ ln(0.5) -0.25a/b - 0.75*ln(1.5) + 0.5*ln(T) ]
#              +  exp[ (2/3)ln(T) - a/(3b) + a/(12b) + 0.25*ln(1.5) - (1/6)ln(T) ]
#            = T^0.5 * 1.5^(3/4) * exp[-0.25a/b].
# 
# So a/(4b) = ln(1.5^(3/4) * sqrt(T) / gamma)
#    b = a / [4 ln(1.5^(3/4) * sqrt(T) / gamma) ].
#    s = (a/4) * (1 + [1/ln(1.5^0.75 * sqrt(T) / gamma)]*(0.75*ln(1.5) - 0.5*ln(T))
#    s = (a/4) * (1 - FRAC)
# 
# FRAC = 0.5*ln(T) - ln(1.5^0.75)
#        ---------------------------------------
#        0.5ln(T) + ln(1.5^0.75) + ln(1.0/gamma)


