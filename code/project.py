import numpy as np

# assumes vec is a numpy one-dimensional array
# check if its norm exceeds max_norm, if so, "project"
# by renormalizing
def one_norm_project(vec, max_norm):
  norm = sum(list(map(abs, vec)))
  ratio = norm / max_norm
  if ratio > 1.0:
    return vec / ratio
  else:
    return vec

def two_norm_project(vec, max_norm):
  norm = np.linalg.norm(vec)
  ratio = norm / max_norm
  if ratio > 1.0:
    return vec / ratio
  else:
    return vec
