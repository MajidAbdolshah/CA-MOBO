The obtained Pareto solutions for every iteration will be saved in a folder for your consideration.  For changing the proposed functions for optimisation,  check function.py. Accordingly, parameters.py needs to be updated too:

INPUT_DIM = # of input dimensions
OUTPUT_DIM = # of output dimensions

INITIAL = # of initial observations
MAXSAMPLE = 10**6 (parameter for inside optimiser of BO, recommended not to be changed)
COUNTER = Maximum iterations
RPT = # repeating the experiment

LEN_SCALE = initial value of length scale for SE kernel
VARIANCE_ = initial value of sigma for SE kernel

To run the code:
python main.py

To see the obtained results:
python AnalyseMe.py  [#of iterations] [#of re-running experiment]  [boolean value for detailed plot]

Example:
python AnalyseMe.py  500 50 0

** You do not need to stop the optimisation to check out the results. All the obtained results will be saved for every iteration. **
