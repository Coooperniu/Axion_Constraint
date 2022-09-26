#######################################################
###        Likelihood-Ratio Test for ma-ga          ###
###    for a trial project with Prof. Jiji Fan      ###
###              by Cooper Niu, 2022                ###
#######################################################


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

import os
import errno
import sys
import getopt
import warnings
import random
import h5py

import numpy as np
from numpy import pi, sqrt, log, log10, exp, power
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import LinearNDInterpolator as lndi
from tqdm import tqdm
from analysis_main import pltpath


##########################
# auxiliary functions
##########################

def run_lkl_null(nbin = 50, pdir = ''):
    """
    output the contours in ma-ga space from the frequentist likelihood ratio test.
    """

    #--------------------------#
    #      Read Chain file     #
    #--------------------------#

    flgn = True

    directory = pdir
    bins = nbin

    # reading chains
    path = directory+'/chain.h5'
    f = h5py.File(path, 'r')

    f = f['mcmc']
    keys = f.keys()

    pts = np.array(f['chain']) # the points
    pts = pts.reshape(-1, 6)
    
    chi2_tot = np.array(f['log_prob'])
    chi2_tot *= -2
    chi2_tot = chi2_tot.reshape(-1)

    blobs = f['blobs']
    experiments = dict(blobs.dtype.fields).keys()

    each_chi2 = {exper:blobs[exper].reshape(-1) for exper in experiments} 

    del f
    
    bf_chi2, bf_idx = min(chi2_tot), chi2_tot.argmin() 
    each_sum = sum([each_chi2[exper][bf_idx] for exper in experiments]) 
#    print "chi2 best fit: {} = {}".format(bf_chi2, each_sum) # sanity check
    #-----------------#
    #   print info    #
    #-----------------#

    p_file = '[chi best fit]: '+ str(bf_chi2) + '(null)'+'.txt'
    p_file_path = os.path.join(directory, p_file)
    with open(p_file_path, 'w') as file:
        file.write('nothing here')    
    

if __name__ == '__main__':
    warnings.filterwarnings('error', 'overflow encountered')
    warnings.filterwarnings('error', 'invalid value encountered')
    warnings.filterwarnings('ignore')
 
    argv = sys.argv[1:]

    help_msg = 'Usage: python main.py [option] ... [arg] ...'

    try:
        opts, args = getopt.getopt(argv, 'hp:n:b:')
    except getopt.GetoptError:
        raise Exception(help_msg)

    for opt, arg in opts:
        if opt == '-h':
            print(help_msg)
            sys.exit()
        elif opt == '-p':
            pdir = arg
            flg_w = True
        elif opt == '-b':
            nbins = int(arg)
            flg_b = True

    run_lkl_null(nbin = 50, pdir = str(pdir))




