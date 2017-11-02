#!/bin/bash

# create a temp file
here=$(basename $0)
commandfile=$(mktemp $here.XXXXXX)

# prep_simulations.py commandfile datafile alphalist numtrials lambda gamma maxnorm maxsteps param_start param_end
python3 prep_simulations.py "$commandfile" dataset.txt alphalist.txt 20 0.005 0.1 0 1000 1 5

parallel --gnu --max-procs 8 < "$commandfile"
rm "$commandfile"

