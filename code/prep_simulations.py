#!/usr/bin/python3

import os, sys, traceback

# Use 'run_many_ridge.py' for ridge regression
# or 'run_many_logist.py' for logistic regression
run_file_name = "run_many_ridge.py"

usage_str = """
Usage:  python3 prep_simulations.py args

where args are, in order:
  commandfilename  (a unique temporary file)
  datafilename
  alphalistfilename
  numtrials
  lambda
  gamma
  max_norm
  max_steps
  param_start (optional)
  param_end (optional)

Creates the results directory, an about.py file, and
directories for the results.
Writes a list of commands to run that produce the
results into commandfilename.

If param_start and param_end are specified,
only run the simulations for alphalist[param_start], ...,
alphalist[param_end] inclusive.
"""

topdirname = "sim-results/"

#if os.path.exists(topdirname):
#  print(topdirname + " already exists!")
#  exit(0)

try:
  commandname = sys.argv[1]
  dataname = sys.argv[2]
  alphaname = sys.argv[3]
  for fname in [dataname, alphaname]:
    if not os.path.exists(fname):
      print("Could not find " + fname)
      raise Exception()
  with open(dataname) as f:
    n = 0
    for line in f:
      n += 1
    f.seek(0)
    dim = len(f.readline().strip().split()) - 1

  with open(alphaname) as f:
    alphalist = list(map(float, [s.strip() for s in f.readlines()]))
  num_trials = int(sys.argv[4])
  lamb = float(sys.argv[5])
  gamma = float(sys.argv[6])
  max_norm = int(sys.argv[7])
  max_steps = int(sys.argv[8])

  if len(sys.argv) > 10:
    param_start = int(sys.argv[9])
    param_end   = int(sys.argv[10])
  else:
    param_start = 1
    param_end = len(alphalist)

except:
  print(usage_str)
  print("----")
  print(traceback.format_exc())
  exit(0)

os.makedirs(topdirname, exist_ok=True)

if not os.path.exists(topdirname + "about.py"):
  with open(topdirname + "about.py", "w") as f:
    f.write("datafilename = \"" + dataname + "\"\n")
    f.write("n = " + str(n) + "\n")
    f.write("dim = " + str(dim) + "\n")
    f.write("alphalist = " + str(alphalist) + "\n")
    f.write("num_trials = " + str(num_trials) + "\n")
    f.write("lamb = " + str(lamb) + "\n")
    f.write("gamma = " + str(gamma) + "\n")
    f.write("max_norm = " + str(max_norm) + "\n")
    f.write("max_steps = " + str(max_steps) + "\n")

with open(commandname, "w") as f:
  for i,alpha in enumerate(alphalist):
    if i+1 < param_start or i+1 > param_end:
      continue
    outputdirname = topdirname + "param-" + str(i+1)
    f.write(" ".join(["python3", run_file_name, dataname,str(lamb),str(alpha),str(gamma),str(max_norm),str(max_steps), str(num_trials), outputdirname]))
    f.write("\n")

