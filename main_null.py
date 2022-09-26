############################################
###     Run All Combinations of cases     ###
#############################################


import os
import warnings
import getopt
import sys

from axion_main import run_emcee_code
from analysis_main import make_corner
from lkl_ratio import run_lkl_ratio
from lkl_null import run_lkl_null

if __name__ == '__main__':
    warnings.filterwarnings('error', 'overflow encountered')
    warnings.filterwarnings('error', 'invalid value encountered')
    warnings.filterwarnings('ignore')

    argv = sys.argv[1:]

    help_msg = 'Usage: python main.py [option] ... [arg] ...  \n' + \
               'Options and arguments (and corresponding environment variables): \n' + \
               '        -n : nwalkers (int) – The number of walkers in the ensemble. \n' + \
               '        -w : nsteps (int) - The number of steps to run. \n' + \
               '             Iterate sample() for nsteps iterations and return the result. \n' + \
               '        -b : nbins (int) - The number of ma-ga contour bins.'

    try:
        opts, args = getopt.getopt(argv, 'hn:w:b:')
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
        elif opt == '-b':
            nbins = int(arg)
            flg_b = True

    count = 0
#    data_set = ['early', 'late']
    data_set = ['late']
    model = ['False', 'A', 'B', 'C']
    ne_IGM = [1.6, 3.0]
    s_IGM = [0.1, 1., 10.]
    mu = [-1, 1]

    # run over all possible combinations of variable parameters 
    for d in data_set:
        print("Run No. [", count, ']')
        print('##################################################')
        count += 1
                        
        output_dir_pos = run_emcee_code(data_combo = d, 
                                        ICM_magnetic_model = 'False', 
                                        ne_IGM = 1.6 * 1e-08, 
                                        s_IGM = 1., 
                                        mu = 1,
                                        nwalkers = nwalkers, 
                                        nsteps = nsteps)                        
                  
        run_lkl_null(nbin = nbins, pdir = str(output_dir_pos))
                        

    print(count)
    print("This Super Big Run is finished! Finally!")






