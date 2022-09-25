###############################################################
###                Calculate the Chi Square                 ###
###         for a trial project with Prof. Jiji Fan         ###
###                   by Cooper Niu, 2022                   ###
###############################################################



#=================#
# Import Packages #
#=================#

import numpy as np
import scipy.linalg as la
from numpy import pi, sqrt, log, log10, exp, power
#from cosmo import H_at_z, tau_at_z, dA_at_z, muLCDM, LumMod, ADDMod

from IGM import H, Comove_D 
from observables import Hubble, ADD, mu_mod, delta_mu, ADD_mod
from load_data import check_range



#=============================#
# Initialization of Constants #
#=============================#

c0 = 299792458. 
rs_drag_mean = 147.09
rs_drag_sig = 0.26
h_TD = 0.745
h_TD_sig = 0.0585



#=============================#
#          Chi Square         #
#=============================#

def chi2_SH0ES(M0, data=None):

    name, SN, _ , Cepheid, _ , _, M_sigma = data

    chi2 = 0.

    for i in range(len(name)):
        chi2 += (SN[i] - M0 - Cepheid[i])**2 / M_sigma[i]**2

    return chi2



def chi2_BAO_DR12(pars, data=None):

    (Omega_L, h0, rs) = pars
    BAO_rsfid, BAO_z, BAO_dM, BAO_Hz, BAO_cov, BAO_inv_cov = data
 
    chi2 = 0.
    data_array = np.array([], 'float64')
 
    for i, z in enumerate(BAO_z):
 
        dM = Comove_D(z, h0, Omega_L)
        Hz = h0 * 100. * H(Omega_L, z)
 
        dM_theory = dM / rs * BAO_rsfid
        Hz_theory = Hz * rs / BAO_rsfid
  
        delta_dM = dM_theory - BAO_dM[i]
        delta_Hz = Hz_theory - BAO_Hz[i]
 
        data_array = np.append(data_array, delta_dM)
        data_array = np.append(data_array, delta_Hz)
 
    chi2 += np.dot(np.dot(data_array, BAO_inv_cov), data_array)
 
    return chi2


def chi2_BAO_lowz(pars, data=None):

    (Omega_L, h0, rs) = pars
    name, BAO_z, BAO_rs_dV, BAO_sigma, BAO_type = data
 
    chi2 = 0.
    for i, z in enumerate(BAO_z):
        dm = ADD(z, h0, Omega_L)
        dr = z / Hubble(z, h0, Omega_L)
        dv = (dm**2 * (1 + z)**2 * dr)**(1./3.)
 
        if BAO_type[i] == 3:
            ratio = dv / rs
        if BAO_type[i] == 7:
            ratio = rs / dv
        chi2 += ((ratio - BAO_rs_dV[i]) / BAO_sigma[i])**2
 
    return chi2



def chi2_Pantheon(pars, data=None, **kwargs):

    (ma, ga, OmL, h0, M0) = pars
    name, pan_z, pan_zhel, pan_dz, pan_dm, pan_dmb, pan_cov = data

    chi2 = 0.
    residuals = []

    for i in range(len(name)):

        change = delta_mu(ma, ga, pan_z[i], h=h0, Omega_L=OmL, **kwargs)
        residuals.append(mu_mod(pan_z[i], h0, OmL) - pan_dm[i] + M0 - change)

    L_residuals = la.solve_triangular(pan_cov, residuals, lower=True, check_finite=False)
    chi2 = np.dot(L_residuals, L_residuals)

    return chi2



def chi2_h0(h0):

    chi2 = 0.
    chi2 += (h0 - h_TD)**2 / h_TD_sig**2

    return chi2



def chi2_rs(rs):

    chi2 = 0.
    chi2 += (rs - rs_drag_mean)**2 / rs_drag_sig**2

    return chi2



def chi2_clusters(pars, data=None, err_correct=True, fixed_Rvir=False, **kwargs):

    (ma, ga, Omega_L, h0) = pars
    name, z_cls, DA_cls, err_cls, asymm_cls, ne0_cls, beta_cls, rc_out_cls, f_cls, rc_in_cls, Rvir_cls = data

    chi2 = 0.
    residuals = []

    for i in range(len(name)):

        if fixed_Rvir:
            r_up = 1800.
        else:
            r_up = Rvir_cls[i] 

        DA_theory = ADD(z_cls[i], h0, Omega_L) * ADD_mod(ma, ga, z_cls[i], h0, Omega_L,
                                                     ne0 = ne0_cls[i],
                                                     rc_outer = rc_out_cls[i],
                                                     beta_outer = beta_cls[i],
                                                     f_inner = f_cls[i],
                                                     rc_inner = rc_in_cls[i],
                                                     beta_inner = beta_cls[i],
                                                     r_up = r_up,
                                                     **kwargs)

        residuals.append(DA_cls[i] - DA_theory)       

    factor = 1.

    if err_correct:
        factor += -2.*asymm_cls * (residuals/err_cls) + 5.*asymm_cls**2. * (residuals/err_cls)**2.
     
    
    terms = ((residuals / err_cls)**2.) * factor
    chi2 = terms.sum()

    return chi2




#==============================#
#       Total Chi Square       #
#==============================#

def total_chi2(pars,
           key=None,
           param=None,
           use_SH0ES=False, sh0es_data=None,
           use_BAO_DR12=False, bao_dr12_data=None,
           use_BAO_lowz=False, bao_lowz_data=None,
           use_Pantheon=False, pan_data=None, pan_kwargs=None,
           use_TDCOSMO=False, use_early=False,
           use_clusters=False, clusters_data=None, clusters_kwargs=None,
           err_correct=True,
           fixed_Rvir=False,
           verbose=False):

    """
    Computes the total likelihood, as well as that for each experiment
    """

    variable = {}
 
    for i in range(len(key)):
        variable[key[i]] = pars[i]

    ma = 10**variable['logma']
    ga = 10**variable['logga']
    OmL = variable['OmL']
    h0 = variable['h0']
 
    if use_Pantheon or use_SH0ES:
        M0 = variable['M0']
    if use_BAO_DR12:
        rs = variable['rs']
 
    count = sum([use_SH0ES, use_Pantheon, use_TDCOSMO, use_early, use_BAO_DR12, use_BAO_lowz, use_clusters])
 
    chi2_list = []
    total_chi2 = 0.
    out_range = False
 
    if not check_range(pars, key, param): 
        chi2 = 0

        if use_SH0ES:

            chi2 = chi2_SH0ES(M0, data=sh0es_data)
            total_chi2 += chi2
            chi2_list.append(chi2)
#            print('SHOES=%f' % chi2)

        # Pantheon
        if use_Pantheon:

            chi2 = chi2_Pantheon((ma, ga, OmL, h0, M0), data=pan_data, **pan_kwargs)
            total_chi2 += chi2
            chi2_list.append(chi2)
#            print('pantheon=%f' % chi2)
 
        # other H0 experiments
        if use_TDCOSMO:
 
            chi2 = chi2_h0(h0)
            total_chi2 += chi2
            chi2_list.append(chi2)
#            print('TDCOSMO=%f' % chi2)

        if use_early:
 
            chi2 = chi2_rs(rs)
            total_chi2 += chi2
            chi2_list.append(chi2)
#            print('early=%f' % chi2)
 
        # BOSS DR12
        if use_BAO_DR12:
 
            chi2 = chi2_BAO_DR12((OmL, h0, rs), data=bao_dr12_data)
            total_chi2 += chi2
            chi2_list.append(chi2)
#            print('boss=%f' % chi2)

        # BAOlowz (6DFs + BOSS DR7 MGS, called smallz in MontePython)
        if use_BAO_lowz:

            chi2 = chi2_BAO_lowz((OmL, h0, rs), data=bao_lowz_data)
            total_chi2 += chi2
            chi2_list.append(chi2)
#            print('bao=%f' % chi2)
 
        # clusters
        if use_clusters:

            chi2 = chi2_clusters((ma, ga, OmL, h0), data=clusters_data, err_correct=err_correct, fixed_Rvir=fixed_Rvir, **clusters_kwargs)
            total_chi2 += chi2
            chi2_list.append(chi2)
#            print('clusters=%f' % chi2)

    else:
        total_chi2 = np.inf
        chi2_list = [np.inf]*count
#        print("out of range... chi2 = np.inf")
 
    if np.isnan(total_chi2) or np.any(np.isnan(chi2_list)):
        total_chi2 = np.inf
        chi2_list = [np.inf]*count
 
    res = -1./2.* total_chi2

    chi2_list.insert(0, res)
    chi2_list = tuple(chi2_list)

    return chi2_list

"""
# just a test
#

from load_data import load_BAO_DR12, load_BAO_lowz, load_sh0es, load_pantheon, load_clusters, load_param, set_param_default

input_path = "/Users/cooper/Axion/final_2/inputs/example.param"
#input_path = "/Users/cooper/Axion/final_2/example.param"
data_path = "/Users/cooper/Axion/TProject/datasets/"
#
param, var = load_param(input_path)
#
#print(param)
#print(var)
#
pan_kwargs, clusters_kwargs = set_param_default(param, var)
#
if param['use_SH0ES'] is True:
    sh0es_data = load_sh0es(data_path)
else:
    sh0es_data = None
#
if param['use_Pantheon'] is True:
    pan_data = load_pantheon(data_path)
else:
    pan_data = None
#
if param['use_BAO_DR12'] is True:
    bao_dr12_data = load_BAO_DR12(data_path)
else:
    bao_dr12_data = None
#
if param['use_BAO_lowz'] is True:
    bao_lowz_data = load_BAO_lowz(data_path)
else:
    bao_lowz_data = None
#
if param['use_clusters'] is True:
    clusters_data = load_clusters(data_path)
else:
    clusters_data = None
#
hsh = total_chi2((0.6847, 0.73, -14, -11, -19.3, 147.78), key=var, param=param,
                           use_SH0ES=param['use_SH0ES'], sh0es_data=sh0es_data,
                           use_BAO_DR12=param['use_BAO_DR12'], bao_dr12_data=bao_dr12_data,
                           use_BAO_lowz=param['use_BAO_lowz'], bao_lowz_data=bao_lowz_data,
                           use_Pantheon=param['use_Pantheon'], pan_data=pan_data, pan_kwargs=pan_kwargs,
                           use_TDCOSMO=param['use_TDCOSMO'],
                           use_early=param['use_early'],
                           use_clusters=param['use_clusters'], clusters_data=clusters_data,
                           err_correct=param['err_correct'], 
                           fixed_Rvir= param['fixed_Rvir'], clusters_kwargs=clusters_kwargs,
                           verbose=param['verbose'])
#
print(hsh)
#
#print(chi_square((0.6847, 0.73, -14, -11, -19.3, 147.78)))


"""
