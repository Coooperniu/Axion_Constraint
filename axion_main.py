##########################################################
###                  Axion Main Function                ###
###         for a trial project with Prof. Jiji Fan     ###
###                   by Cooper Niu, 2022               ###
###########################################################


#=================#
# Import Packages #
#=================#

import matplotlib.pyplot as plt
from datetime import datetime

import os
import errno
import emcee
import sys
import getopt
import warnings
import random

import numpy as np

from numpy import pi, sqrt, log, log10, exp, power
from contextlib import closing



# -----------------MAIN-CALL---------------------------------------------

if __name__ == '__main__':
    
    argv = sys.argv[1:]

    opts, args = getopt.getopt(argv, 'hN:o:L:i:w:')

    flgN = False
    flgo = False
    flgL = False
    flgi = False
    flgw = False
    for opt, arg in opts:
        if opt == '-h':
            raise Exception(help_msg)
        elif opt == '-N':
            chainslength = arg
            flgN = True
        elif opt == '-o':
            directory = arg
            flgo = True
        elif opt == '-L':
            dir_lkl = arg
            flgL = True
        elif opt == '-i':
            path_of_param = arg
            flgi = True
        elif opt == '-w':
            number_of_walkers = int(arg)
            flgw = True

    param, var = load_param(path_of_param)






