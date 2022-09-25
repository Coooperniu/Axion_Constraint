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
from ag_conversion import m_gamma
from chi_square import total_chi2


#=====================================#
#            MAIN FUNCTION            #
#=====================================#

def run_emcee_code(data_combo = 'late', ICM_magnetic_model  = 'A', ne_IGM = 1.6e-08, s_IGM = 1., mu = -1., 
                   nwalkers = 100, nsteps = 100):
    """
    Run the emcee program, scanning over ma, ga parameter space
    """

    warnings.filterwarnings('error', 'overflow encountered')
    warnings.filterwarnings('error', 'invalid value encountered')
    warnings.filterwarnings('ignore')
    
    
    #--------------------------#
    #   Initialize Directory   #
    #--------------------------#

    param_path = 'example.param'
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

    #-----------------#
    #   print info    #
    #-----------------#

    param_file = '['+str(data_combo)+','+str(ICM_magnetic_model)+','+str(ne_IGM)+','+str(mu)+'].txt' 
    param_file_path = os.path.join(output_path, param_file)
    with open(param_file_path, 'w') as f:
        f.write('['+str(data_combo)+','+str(ICM_magnetic_model)+','+str(ne_IGM)+','+str(mu)+']')

    
    #------------------------#
    #    Initialize Param    #
    #------------------------#

    # fill in pan_kwargs and clusters_kwargs used in the probability function

    try:
        err_correct = param['err_correct']
    except KeyError:
        err_correct = True

    try:
        smoothed_IGM = param['smoothed_IGM']
    except KeyError:
        smoothed_IGM = False

    try:
        method_IGM = param['method_IGM']
    except KeyError:
        method_IGM = 'simps'

    try:
        Nz_IGM = param['Nz_IGM']
    except KeyError:
        Nz_IGM = 501

    try:
        prob_func_IGM = param['prob_func_IGM']
    except KeyError:
        prob_func_IGM = 'norm_log'

    try:
        omegaSN = param['omegaSN [eV]']
    except KeyError:
        omegaSN = 1.

    try:
        B_IGM = param['B_IGM [nG]']
    except KeyError:
        B_IGM = 1.

#    try:
#        ne_IGM = param['ne_IGM [1/cm3]']
#    except KeyError:
#        ne_IGM = 6.e-8

#    try:
#        s_IGM = param['s_IGM [Mpc]']
#    except KeyError:
#        s_IGM = 1.

#    try:
#        ICM_effect = param['ICM_effect']
#    except KeyError:
#        ICM_effect = False

    try:
        smoothed_ICM = param['smoothed_ICM']
    except KeyError:
        smoothed_ICM = True

    try:
        method_ICM = param['method_ICM']
    except KeyError:
        method_ICM = 'product'

    try:
        prob_func_ICM = param['prob_func_ICM']
    except KeyError:
        prob_func_ICM = 'norm_log'

    try:
        Nr_ICM = param['Nr_ICM']
    except KeyError:
        Nr_ICM = 501

    try:
        los_method = param['los_method']
    except KeyError:
        los_method = 'quad'

    try:
        los_Nr = param['los_Nr']
    except KeyError:
        los_Nr = 501

    try:
        omega_Xrays = param['omegaX [keV]']*1.e3
    except KeyError:
        omega_Xrays = 1.e4

    try:
        omega_CMB = param['omegaCMB [eV]']
    except KeyError:
        omega_CMB = 2.4e-4

    try:
        fixed_Rvir = param['fixed_Rvir']
    except KeyError:
        fixed_Rvir = False

    try:
        L_ICM = param['L_ICM [kpc]']
    except KeyError:
        L_ICM = 6.08

    if ICM_magnetic_model == 'A':

        r_low = 10.
        B_ref = 25.
        r_ref = 0.
        eta = 0.7
        ICM_effect = True        

    elif ICM_magnetic_model == 'B':

        r_low = 0.
        B_ref = 7.5
        r_ref = 25.
        eta = 0.5
        ICM_effect = True

    elif ICM_magnetic_model == 'C':

        r_low = 0.
        B_ref = 4.7
        r_ref = 0.
        eta = 0.5
        ICM_effect = True

    elif ICM_magnetic_model == 'False':
        
        ICM_effect = False


    #-----------------#
    #    Load Data    #
    #-----------------#

    experiments = []
    
    if data_combo == 'early':

        use_Pantheon = True
        use_BAO_DR12 = True
        use_BAO_lowz = True
        use_clusters = True
        use_early = True
        use_SH0ES = False
        use_TDCOSMO = False

    elif data_combo == 'late':

        use_Pantheon = True
        use_BAO_DR12 = True
        use_BAO_lowz = True
        use_clusters = True
        use_early = False
        use_SH0ES = True
        use_TDCOSMO = True

    else:
        print("You need to choose between EARLY and LATE data combinations!")


    if use_SH0ES is True:
        sh0es_data = load_sh0es(data_path)
        experiments.append('sh0es')
    else:
        sh0es_data = None

    if use_Pantheon is True:
        pan_data = load_pantheon(data_path)
        experiments.append('pantheon')
    else:
        pan_data = None

    if use_BAO_DR12 is True:
        bao_dr12_data = load_BAO_DR12(data_path)
        experiments.append('bao_dr12')
    else:
        bao_dr12_data = None

    if use_BAO_lowz is True:
        bao_lowz_data = load_BAO_lowz(data_path)        
        experiments.append('bao_lowz')
    else:
        bao_lowz_data = None

    if use_clusters is True:
        clusters_data = load_clusters(data_path)
        experiments.append('clusters')
    else:
        clusters_data = None

    if use_early is True:
        experiments.append('planck')

    if use_TDCOSMO is True:
        experiments.append('tdcosmo')



    # print out the info about the run
    print('Signal Strength:', mu )
    print('Experiments: ', experiments)
    print('ICM Magnetic Model: ', ICM_magnetic_model)
    print('ICM Effect: ', ICM_effect)
    print('IGM electron number density: ', ne_IGM)
    print('IGM magnetic domain: ', s_IGM)



    #----------------------#
    #  Keyword Dictionary  #
    #----------------------#

    if use_Pantheon:

        pan_kwargs = {'B':B_IGM,
                      'mg':m_gamma(ne_IGM),
                      's':s_IGM,
                      'omega':omegaSN,
                      'axion_ini_frac':0.,
                      'smoothed':smoothed_IGM,
                      'method':method_IGM,
                      'prob_func':prob_func_IGM,
                      'Nz':Nz_IGM,
                      'mu':mu}
    else:
        pan_kwargs = None

    if use_clusters:

        clusters_kwargs = {'omega_Xrays':omega_Xrays,
                           'omega_CMB':omega_CMB,
                           's_IGM':s_IGM,
                           'B_IGM':B_IGM,
                           'mg_IGM':m_gamma(ne_IGM),
                           'smoothed_IGM':smoothed_IGM,
                           'method_IGM':method_IGM,
                           'prob_func_IGM':prob_func_IGM,
                           'Nz_IGM':Nz_IGM,
                           'ICM_effect':ICM_effect,
                           'r_low':r_low,
                           'L':L_ICM,
                           'smoothed_ICM':smoothed_ICM,
                           'method_ICM':method_ICM,
                           'prob_func_ICM':prob_func_ICM,
                           'Nr_ICM':Nr_ICM,
                           'los_method':los_method,
                           'los_Nr':los_Nr,
                           'mu':mu,
                           'B_ref':B_ref,
                           'r_ref':r_ref,
                           'eta':eta}

    else:
        clusters_kwargs = None


    # 'total_chi2' function arguments dict
    ln_kwargs = {'key': var,
                 'param':param,
                 'use_SH0ES':use_SH0ES,
                 'sh0es_data':sh0es_data,
                 'use_BAO_DR12':use_BAO_DR12,
                 'bao_dr12_data':bao_dr12_data,
                 'use_BAO_lowz':use_BAO_lowz,
                 'bao_lowz_data':bao_lowz_data,
                 'use_Pantheon':use_Pantheon,
                 'pan_data':pan_data,
                 'pan_kwargs':pan_kwargs,
                 'use_TDCOSMO':use_TDCOSMO,
                 'use_early':use_early,
                 'use_clusters':use_clusters,
                 'clusters_data':clusters_data,
                 'err_correct':err_correct,
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

#    print("p0_mean:", p0_mean)
#    print("p0_sigma:", p0_sigma)

    for i in range(ndim):
        p0_array = np.random.normal(p0_mean[i], p0_sigma[i], nwalkers)
        p0.append(p0_array)
    p0 = np.array(p0).T

    # Set up the backend
    chain_path = os.path.join(output_path, "chain.h5")
    backend = emcee.backends.HDFBackend(chain_path)
    backend.reset(nwalkers, ndim)

    from multiprocessing import Pool
    
    # Run the emcee sampler
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        total_chi2,
                                        backend = backend,
                                        pool = pool,
                                        blobs_dtype = [(i, float) for i in experiments],
                                        kwargs = ln_kwargs)
        sampler.reset()
        result = sampler.run_mcmc(p0, nsteps, store = True, progress = True)
        pool.terminate()
       # print('multiprocessing is at play!')

#    print("Length of acceptance fraction: ", len(sampler.acceptance_fraction)) 
#    print("Acceptance fraction: ", sampler.acceptance_fraction) 
    print("Mean acceptance fraction: {0:.3f}".format(
         np.mean(sampler.acceptance_fraction)))

#    print("Mean autocorrelation time: {0:.3f} steps".format(
#         np.mean(sampler.get_autocorr_time())))

    return output_path

#if __name__ == '__main__':
#    run_emcee_code()



