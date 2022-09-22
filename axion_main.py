###########################################################
###                  Axion Main Function                ###
###         for a trial project with Prof. Jiji Fan     ###
###                   by Cooper Niu, 2022               ###
###########################################################


#=================#
# Import Packages #
#=================#

import os
import errno
import emcee
import sys
import getopt
import warnings
import random
import glob
import numpy as np
import time

from load_data import load_BAO_DR12, load_BAO_lowz, load_sh0es, load_pantheon, load_clusters, load_param, set_param_default
from chi_square import total_chi2


#=====================================#
#            MAIN FUNCTION            #
#=====================================#

if __name__ == '__main__':
    warnings.filterwarnings('error', 'overflow encountered')
    warnings.filterwarnings('error', 'invalid value encountered')
    warnings.filterwarnings('ignore')
    
    argv = sys.argv[1:]

    help_msg = 'Usage: python axion_main.py [option] ... [arg] ...  \n' + \
               'Options and arguments (and corresponding environment variables): \n' + \
               '        -n : nwalkers (int) – The number of walkers in the ensemble. \n' + \
               '        -w : nsteps (int) - The number of steps to run. \n' + \
               '             Iterate sample() for nsteps iterations and return the result. \n' + \
               '        -i : parameter file (str) - The directory of the input .paran file.'

    try:
        opts, args = getopt.getopt(argv, 'hn:o:d:i:w:')
    except getopt.GetoptError:
        raise Exception(help_msg)   

    flg_i = False 
    flg_n = False
    flg_w = False

    for opt, arg in opts:
        if opt == '-h':
            print(help_msg)
            sys.exit()
        elif opt == '-n':
            nstep = int(arg)
            flg_n = True
        elif opt == '-i':
            param_path = arg
            flg_i = True
        elif opt == '-w':
            nwalkers = int(arg)
            flg_w = True
    
    output_path = 'output'+ time.strftime('-%Y-%m-%d-%H-%M')
    data_path = 'datasets/'    
 
    # make new output directory 
    try:
        os.makedirs(output_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # read off the .param file inputs
    if os.path.exists(os.path.join(output_path, 'log.param')):
        param_path = os.path.join(output_path, 'log.param')
        param, var = load_param(param_path)

    else:
        param, var = load_param(param_path)
        from shutil import copyfile
        copyfile(param_path, os.path.join(output_path, 'log.param'))
    
    # fill in pan_kwargs and clusters_kwargs used in the probability function
    pan_kwargs, clusters_kwargs = set_param_default(param, var)    

    # load data
    experiments = []
 
    if param['use_SH0ES'] is True:
        sh0es_data = load_sh0es(data_path)
        experiments.append('sh0es')
    else:
        sh0es_data = None

    if param['use_Pantheon'] is True:
        pan_data = load_pantheon(data_path)
        experiments.append('pantheon')
    else:
        pan_data = None

    if param['use_BAO_DR12'] is True:
        bao_dr12_data = load_BAO_DR12(data_path)
        experiments.append('bao_dr12')
    else:
        bao_dr12_data = None

    if param['use_BAO_lowz'] is True:
        bao_lowz_data = load_BAO_lowz(data_path)        
        experiments.append('bao_lowz')
    else:
        bao_lowz_data = None

    if param['use_clusters'] is True:
        clusters_data = load_clusters(data_path)
        experiments.append('clusters')
    else:
        clusters_data = None

    if param['use_early'] is True:
        experiments.append('planck')

    if param['use_TDCOSMO'] is True:
        experiments.append('tdcosmo')


    # 'total_chi2' function arguments dict
    ln_kwargs = {'key': var,
                 'param':param,
                 'use_SH0ES':param['use_SH0ES'],
                 'sh0es_data':sh0es_data,
                 'use_BAO_DR12':param['use_BAO_DR12'],
                 'bao_dr12_data':bao_dr12_data,
                 'use_BAO_lowz':param['use_BAO_lowz'],
                 'bao_lowz_data':bao_lowz_data,
                 'use_Pantheon':param['use_Pantheon'],
                 'pan_data':pan_data,
                 'pan_kwargs':pan_kwargs,
                 'use_TDCOSMO':param['use_TDCOSMO'],
                 'use_early':param['use_early'],
                 'use_clusters':param['use_clusters'],
                 'clusters_data':clusters_data,
                 'err_correct':param['err_correct'],
                 'fixed_Rvir':param['fixed_Rvir'],
                 'clusters_kwargs':clusters_kwargs,
                 'verbose':param['verbose']}

    # initialize the sets of starting points p0
    p0 = []
    p0_mean = []
    p0_sigma = []
   
    for key in var:
        p0_mean.append(param[key+'_mean'])
        p0_sigma.append(param[key+'_sig'])

    ndim = len(p0_mean)

    print("p0_mean:", p0_mean)
    print("p0_sigma:", p0_sigma)

    for i in range(ndim):
        p0_array = np.random.normal(p0_mean[i], p0_sigma[i], nwalkers)
        p0.append(p0_array)
    p0 = np.array(p0).T

    # Set up the backend
    chain_path = os.path.join(output_path, "chain.h5")
    backend = emcee.backends.HDFBackend(chain_path)
    backend.reset(nwalkers, ndim)

    # Print out the guidance information    
    #guide_info = "The function generates a folder (" + output_path + "), in which you can find a log.param file and a chain.file \n" + \
    #             "Please use these two files to continue the axion analysis."
    #print(guide_info) 

    from multiprocessing import Pool
    print(len(experiments))
    
    # Run the emcee sampler
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        lnprob,
                                        backend = backend,
                                        pool = pool,
                                        blobs_dtype = [(i, float) for i in experiments],
                                        kwargs = ln_kwargs)
        sampler.reset()
        result = sampler.run_mcmc(p0, nstep, store = True, progress = True)
        pool.terminate()
       # print('multiprocessing is at play!')

#    print("Length of acceptance fraction: ", len(sampler.acceptance_fraction)) 
#    print("Acceptance fraction: ", sampler.acceptance_fraction) 
    print("Mean acceptance fraction: {0:.3f}".format(
         np.mean(sampler.acceptance_fraction)))

    print("Mean autocorrelation time: {0:.3f} steps".format(
         np.mean(sampler.get_autocorr_time())))



