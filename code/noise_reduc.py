# Noise reduction method
#
# utilizing "Gradual Release of Sensitive Data under Differential Privacy"
# by Fragkiskos Koufogiannis, Shuo Han, and George J. Pappas

# Background:
# We wish to release a function f of a private database, say D.
# The true output is a numpy array M = f(D).
# Each entry of M has "sensitivity" s meaning that if D changes to a
# neighboring database D', then the corresponding entry of M' = f(D')
# changes by at most s.
#
# We use differential privacy to release private versions of M.
# The output is a sequence of versions of M that become more accurate
# and less private.
#
# Given M, sensitivity, and a INCREASING list of epsilons eps_1, ..., eps_T
# 1. Construct a random walk starting from M, so that at step t, the {i,j}
#    entry is distributed as M_{ij} + Laplace(1 / eps_{T-t}).
# 2. Return the list of resulting matrices IN REVERSE ORDER of the walk,
#    i.e. from most noisy (most private) to least noisy (least private).
# 3. In particular, releasing all of the first t matrices in the list
#    is eps_t private, for each t, because:
#    - releasing the t-th matrix in the list is eps_t private
#    - all previous matrices are post-processings of this (by the random walk)

import numpy as np


# Helper function
# filter by returning each entry of matrix independently with probability 1 - prob
# and 0 with probability prob
def do_filt(prob, matrix):
  f = lambda x: 0 if np.random.random() <= prob else x
  return np.vectorize(f)(matrix)

# Main function
# Input:
#   numpy array M whose privacy is to be protected
#   sensitivity of the each entry of M
#   a INCREASING list of epsilons
#
# Output: a list of Mhat matrices approximating M, where releasing the first t
# matrices is eps_t private
def gen_list(M, sensitivity, eps_list):
  try:
    steps = len(eps_list)
  except:
    steps = eps_list.shape[0]
  shape = M.shape
  rev_eps_list = eps_list[::-1]
  noise_list = [np.random.laplace(scale=sensitivity/eps, size=shape) for eps in rev_eps_list]

  # first step, just add the noise to M
  walk = [M + noise_list[0]]

  # other steps, add the noise of each entry with only a certain probability
  filt_probs = [ (rev_eps_list[j] / rev_eps_list[j-1])**2 for j in range(1, steps) ]
  walk_steps = [do_filt(p, noise_list[1+j]) for j,p in enumerate(filt_probs)]

  for j in range(1, len(rev_eps_list)):
    walk.append(walk[j-1] + walk_steps[j-1])
  walk.reverse()
  return np.array(walk)


