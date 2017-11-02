Code for paper "Accuracy First: Selecting a Differentially Private
                Level for Accuracy-Constrained ERM"
by Ligett, Neel, Roth, Waggoner, Wu

https://arxiv.org/abs/1705.10829


---------------------------------------------------------------
-- Disclaimer

This code is used for simulations of the performance of
differentially private algorithms, but should not be used in practice
to protect actual sensitive data! The theorems are proved for true
randomness and real numbers, while the code uses python's internal
random generators and floating-point numbers.


---------------------------------------------------------------
-- Requirements
1.  python3 with the numpy, matplotlib, and scikit-learn libraries.

2.  (optional) Linux, bash, and the GNU parallel utility
    available from most repositories.
    This is not mandatory, it's just that a bash script is
    used for running a whole bunch of experiments at once
    and in parallel.


---------------------------------------------------------------
-- Usage

Navigate into the code/ directory to run the code.

You will need a dataset file in plain text.
Each row of the file is a data point.
It should contain d+1 space-separated numbers
(for some d) where the first d are "x" and the last is "y".
It is assumed that the L1-norm of each x is at most 1,
and each |y| <= 1.
For logistic regression, each y should be plus or minus 1.
See data/ directories for downloading the datasets used in the
paper and processing them into this format.

You can run a single experiment at a time and print the output,
or run a set of experiments and save the outputs into folders.


---------------------------------------------------------------
-- To run a single experiment for a given data set and parameters:
     $ python3 run_ridge.py [args]
   OR
     $ python3 run_logist.py [args]
Run them with no arguments for help on the args.


---------------------------------------------------------------
-- To run a set of experiments:

1. You should have a dataset file and also a file with a list of
   the alpha parameters to try, called alphalist.txt.
   E.g. you can edit 'gen_alphalist.py' to your liking and then run


     $ gen_alphalist.py > alphalist.txt

2. Edit the file 'run_sims.sh' to set all the parameters to your
   liking. Also edit the top of the file 'prep_simulations.sh'
   to rename the variable 'run_file_name'.
   It should be "run_many_logist.py" if you want logistic regression
   or "run_many_ridge.py" if you want ridge regression.

3. Execute the following (full explanation of what it does below):

     $ ./run_sims.sh

4. Execute the following to read the results, print some output about
   them, and produce some plots.

     $ python3 collect_results_ridge.py sims-results/


----------------
About run_sims.sh:
This will create a folder sims-results/ and run a bunch of simulations
writing the results into that folder, along with an 'about.py'
file that specifies what all the parameters were.

It does the following:

   a. Runs python3 prep_simulations.py [args]
      which creates the folder sim-results/
      and writes about.py into it.
      Also writes a list of commands to a temporary file.

   b. Invokes GNU parallel to run the commands in parallel.
      WARNING: for large datasets, may use up all your RAM and
      crash your computer! Use --max-procs to limit the number
      of commands to run simultaneously.
      Each command that is run is of form in step c.
      
   c. python3 run_many_ridge.py [args]
      Runs num_trials experiments for the given parameters,
      writing the outputs into sim-results/param-i/ for the
      given i.



