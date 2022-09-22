###########################################################
###                  Axion Chain Analysis               ###
###         for a trial project with Prof. Jiji Fan     ###
###                   by Cooper Niu, 2022               ###
###########################################################


#=================#
# Import Packages #
#=================#

import matplotlib.pyplot as plt
import os
import errno
import emcee
import sys
import getopt
import warnings
import random
import h5py
import numpy as np
import corner

from load_data import load_param, set_param_default
from emcee.autocorr import AutocorrError


def pltpath(dir, head='', ext='.pdf'):
    run_name = str(dir).rstrip('/')
    run_name = run_name.split('/')[-1]

    if bool(head):
        return head + '_' + run_name + ext
    else:
        return 'corner_' + run_name + '.pdf'

#=====================================#
#            MAIN FUNCTION            #
#=====================================#


if __name__ == '__main__':
    warnings.filterwarnings('error', 'overflow encountered')
    warnings.filterwarnings('error', 'invalid value encountered')
    warnings.filterwarnings('ignore')
     
    argv = sys.argv[1:]
      
    help_msg = 'Usage: python axion_main.py [option] ... [arg] ...  \n' + \
               '        -i : output directory (str) - The directory folder where the chains are saved.'

    try:
        opts, args = getopt.getopt(argv, 'hi:')
    except getopt.GetoptError:
        raise Exception(help_msg)

    flg_i = False

    for opt, arg in opts:
        if opt == '-h':
            print(help_msg)
            sys.exit()
        elif opt == '-i':
            directory = arg
            flg_i = True

    log_path = os.path.join(directory, "chain.h5")
    samples = emcee.backends.HDFBackend(log_path, read_only=True)
    try:
        tau = samples.get_autocorr_time()
        print('auto correlation time = %s' % tau)
    except AutocorrError as e:
        print('%s' % e)
        tau = e.tau
        print('setting correlation time to the current estimate.')
    
    # use auto-correlation time to estimate burnin here
    burnin = int(2*np.max(tau))
    thin = int(0.5*np.min(tau))
    samples = samples.get_chain(
        discard=burnin, flat=True, thin=thin)
    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("flat chain shape: {0}".format(samples.shape))
    try:
        all_samples = np.append(all_samples, samples, axis=0)
    except:
        all_samples = samples

    param_path = os.path.join(directory, 'log.param')
    param, var = load_param(param_path)


    pdim = len(var)
    mean = np.mean(samples, axis=0)
    print('mean = %s' % mean)

    #----------------------------#
    #        Corner Plot         #
    #----------------------------#

    plt.figure(0)
    labels = [r"$\Omega_\Lambda$", r"$h$", r"$\log\ m_a$", r"$\log\ g_a$"]
    if 'M0' in var:
        labels.append(r"$M$")
    if 'rs' in var:
        labels.append(r"$r_s^{drag}$")

    figure = corner.corner(samples,
                           labels=labels,
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True,
                           title_kwargs={"fontsize": 12})
    axes = np.array(figure.axes).reshape((pdim, pdim))
    
    plt.savefig(pltpath(directory))

    # focusing on ma-ga
    plt.figure(1)
    reduced_labels = [r"$\log\ m_a$", r"$\log\ g_a$"]
    reduced_samples = samples[:,2:4]
    reduced_dim = len(reduced_labels)

    figure = corner.corner(reduced_samples,
                           labels=reduced_labels,
                           quantiles=[0.16, 0.5, 0.84],
                           color='r', show_titles=True,
                           plot_datapoints=False,
                           plot_density=False,
                           # levels=[1.-np.exp(-(2.)**2 /2.)],
                           levels=[0.95],
                           title_kwargs={"fontsize": 12},
                           hist_kwargs={'color':None})
    axes = np.array(figure.axes).reshape((reduced_dim, reduced_dim))

    p = (figure.axes)[2].collections[0].get_paths()[0]
    v = p.vertices

    # saving the points of the 95% C.R. contour
    np.savetxt(pltpath(directory, head='corner_pts', ext='.txt'), v)

    plt.savefig(pltpath(directory, head='custom'))    





