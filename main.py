#############################################
###     Run All Combinations of cases     ###
#############################################


import os
import warnings
import getopt
import sys

from axion_main import run_emcee_code
from analysis_main import make_corner



if __name__ == '__main__':
    warnings.filterwarnings('error', 'overflow encountered')
    warnings.filterwarnings('error', 'invalid value encountered')
    warnings.filterwarnings('ignore')

    argv = sys.argv[1:]

    help_msg = 'Usage: python main.py [option] ... [arg] ...  \n' + \
               'Options and arguments (and corresponding environment variables): \n' + \
               '        -n : nwalkers (int) – The number of walkers in the ensemble. \n' + \
               '        -w : nsteps (int) - The number of steps to run. \n' + \
               '             Iterate sample() for nsteps iterations and return the result.'

    try:
        opts, args = getopt.getopt(argv, 'hn:o:d:i:w:')
    except getopt.GetoptError:
        raise Exception(help_msg)   

    flg_n = False
    flg_w = False

    for opt, arg in opts:
        if opt == '-h':
            print(help_msg)
            sys.exit()
        elif opt == '-n':
            nsteps = int(arg)
            flg_n = True
        elif opt == '-w':
            nwalkers = int(arg)
            flg_w = True

    count = 0
    data_set = ['early', 'late']
    model = ['A', 'B', 'C', 'False']
    ne_IGM = [1.6, 3.0]
    s_IGM = [0.1, 1., 10.]
    mu = [-1, 1]

    # run over all possible combinations of variable parameters 
    for d in data_set:
        for n in ne_IGM:
            for m in model:
                for i in mu:
                    print("Run No. [", count, ']')
                    print('##################################################')
                    count += 1
                        
                    output_dir = run_emcee_code(data_combo = d, 
                                                ICM_magnetic_model = m, 
                                                ne_IGM = n, 
                                                s_IGM = 1., 
                                                mu = i,
                                                nwalkers = nwalkers, 
                                                nsteps = nsteps)
                        
                    make_corner(output_dir)
                        

    print(count)







