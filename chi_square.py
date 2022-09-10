###############################################################
###                Calculate the Chi Square                 ###
###         for a trial project with Prof. Jiji Fan         ###
###                   by Cooper Niu, 2022                   ###
###############################################################

import numpy as np
import scipy.linalg as la
from numpy import pi, sqrt, log, log10, exp, power

from ag_conversion import m_gamma, k, P_ag, P_survival
from IGM import H, Comove_D, P_igm
from ICM import ne_2beta, B_icm, P_icm
from observables import ADD, D_Lum, mu_mod, eff_mu_mod, delta_mu, P_icm_los, ADD_mod
from load_data import load_BAO_DR12, load_BAO_lowz, load_sh0es, load_pantheon, load_clusters, load_param


#=============================#
# Initialization of Constants #
#=============================#
 
c0 = 299792458.                 # speed of light [m/s]
alpha = 1./137                  # fine-struction constant
me = (0.51099895 * 1.e6)        # [eV] electron mass

#=============================#
# Initialization of Parameter #
#=============================#

rs_drag_mean = 147.09
rs_drag_sig = 0.26
h_TD = 0.745
h_TD_sig = 0.0585
# uncomment below to use fake 1% here
# h_TD_sig = 0.00745


def chi2_sh0es(M, data = None):
    name, SN, _ , Cepheid, _ , _, M_sigma = data

    chi2 = 0.
    length = len(SN)

    for i in range(length):
        chi2 += (SN[i] - M - Cepheid[i])**2 / M_sigma[i]**2

    return chi2
        


def chi2_BAO_DR12(pars, data = None):

    (Omega_L, h, rs) = pars
    
    BAO_rsfid, BAO_z, BAO_dM, BAO_Hz, BAO_cov, BAO_inv_cov = data

    chi2 = 0.
    data_array = np.array([], 'float64')
    
    for i, z in enumerate(BAO_z):
    
        dM = Comove_D(z, h, Omega_L)
        Hz = h * 100. * H(Omega_L, z)
    
        dM_theory = dM / rs * BAO_rsfid
        Hz_theory = Hz * rs / BAO_rsfid
        
        delta_dM = dM_theory - BAO_dM[i]
        delta_Hz = Hz_theory - BAO_Hz[i]

        data_array = np.append(data_array, delta_dM)
        data_array = np.append(data_array, delta_Hz)

    chi2 += np.dot(np.dot(data_array, BAO_inv_cov), data_array)

    return chi2 




def chi2_BAO_lowz(pars, data = None):

    (Omega_L, h, rs) = pars
    name, BAO_z, BAO_rs_dV, BAO_sigma, BAO_type = data

    chi2 = 0
    length = len(name)

    for i, z in enumerate(BAO_z):

        D_H = (c0 * 1.e-3) / (h * 1.e2) # Hubble Distance [Mpc]

        dm = ADD(z, h, Omega_L)
        dr = z / (D_H * H(Omega_L, z)) 
        dv = (dm**2 * dr**2)**(1/3)

        if BAO_type[i] == 3:
            chi2 += (( (dv / rs) - BAO_rs_dV[i]) / BAO_sigma[i])**2
        if BAO_type[i] == 7:
            chi2 += (( (rs / dv) - BAO_rs_dV[i]) / BAO_sigma[i])**2

    return chi2





def chi2_pantheon(pars, data = None, **kwargs):

    (ma, ga, Omega_L, h, M) = pars
    name, pan_z, pan_zhel, pan_dz, pan_dm, pan_dmb, pan_cov = data

    chi2 = 0.
    res = []

    for i in range(len(name)):

        diff = delta_mu(ma, ga, pan_z[i], h = h, Omega_L = Omega_L, **kwargs) 
        res.append(mu_mod(pan_z[i], h, Omega_L) - pan_dm[i] + M - diff)

    L_res = la.solve_triangular(pan_cov, res, lower=True, check_finite=False)
    chi2 = np.dot(L_res, L_res)

    return chi2




def chi2_clusters(pars, data = None, err_correct = True, fixed_Rvir=False, **kwargs):

    (ma, ga, Omega_L, h) = pars
    name, cls_z, cls_DA, cls_err, cls_asymm, cls_ne0, cls_beta, cls_rc_out, cls_f, cls_rc_in, cls_Rvir = data

    chi2 = 0.
    res = []
    
    for i in range(len(name)):
        if fixed_Rvir:
            r_up = 1800. #[kpc]
        else:
            r_up = cls_Rvir[i]

        DA_theory = ADD(cls_z[i], h, Omega_L) * ADD_mod(ma, ga, cls_z[i], h, Omega_L,
                                                        ne0 = cls_ne0[i],
                                                        rc_outer = cls_rc_out[i],
                                                        beta_outer = cls_beta[i],
                                                        f_inner = cls_f[i],
                                                        rc_inner = cls_rc_in[i],
                                                        beta_inner = cls_beta[i],
                                                        r_up = r_up,
                                                        **kwargs)

        res.append(cls_DA[i] - DA_theory)
    
    cor = 1.

    if err_correct:
        cor += -2. * cls_asymm + (res / cls_err) + 5. * cls_asymm**2. * (res / cls_err)**2.

    terms = cor * (res / cls_err)**2
    chi2 = terms.sum()
    
    return chi2


       
def chi2_h0(h):

    chi2 = 0.
    chi2 += (h - h_TD)**2 / h_TD_sig**2

    return chi2



def chi2_rs(rs):

    chi2 =0.
    chi2 += (rs - rs_drag_mean)**2 / rs_drag_sig**2

    return chi2



#======================#
#  Total Chi Square    #
#======================#
def total_chi2(pars,
               key=None,
               use_SH0ES=False, sh0es_data=None,
               use_BAO_DR12=False, bao_dr12_data=None,
               use_BAO_lowz=False, bao_lowz_data=None,
               use_Pantheon=False, pan_data=None, pan_kwargs=None,
               use_TDCOSMO=False, ext_data=None,
               use_early=False, early_data=None,
               use_clusters=False, clusters_data=None, 
               err_correct=True, fixed_Rvir=False,clusters_kwargs=None,
               verbose=False):

    """
    Computes the total likelihood, and that for each experiment
    """
    
    param_dict = {}
    length = int(len(key))

    for i in range(length):
        param_dict[key[i]] = pars[i]

    ma = 10 ** param_dict['logma']
    ga = 10 ** param_dict['logga']
    Omega_L = param_dict['Omega_L']
    h = param_dict['h']
    
    if use_Pantheon:
        M = param_dict['M']
    if use_BAO_DR12:
        rs = param_dict['rs']

    chi2_list = []
    total_chi2 = 0.
   
    if use_SH0ES:
        chi2 = chi2_sh0es(M, data = sh0es_data)
        total_chi2 += chi2
        chi2_list.append(chi2)

    if use_Pantheon:
        chi2 = chi2_pantheon((ma, ga, Omega_L, h, M), data = pan_data, **pan_kwargs)
        total_chi2 += chi2
        chi2_list.append(chi2)
    
    if use_TDCOSMO:
        chi2 = chi2_h0(h)
        total_chi2 += chi2
        chi2_list.append(chi2)

    if use_early:
        chi2 = chi2_r(rs)
        total_chi2 += chi2
        chi2_list.append(chi2)

    if use_BAO_DR12:
        chi2 = chi2_BAO_DR12((Omega_L, h, rs), data = bao_dr12_data)
        total_chi2 += chi2
        chi2_list.append(chi2)

    if use_BAO_lowz:
        chi2 = chi2_BAO_lowz((Omega_L, h, rs), data = bao_lowz_data)
        total_chi2 += chi2
        chi2_list.append(chi2)

    if use_clusters:
        chi2 = chi2_clusters((ma, ga, Omega_L, h), 
                             data = clusters_data,
                             err_correct = err_correct,
                             fixed_Rvir = fixed_Rvir,
                             **clusters_kwargs)

        total_chi2 += chi2
        chi2_list.append(chi2)

    total_lkl = -1./2. * total_chi2
    chi2_list.insert(0, total_lkl)
    result = tuple(chi2_list)
        
    return result 

def chi_square(pars):
    """
    fill in total_chi2 with parameters 
    """
    
    input_path = "/Users/cooper/Axion/test/inputs/neg_sig_example.param"
    data_path = "/Users/cooper/Axion/TProject/datasets/"

    param, var = load_param(input_path)
    
    if param['debug']:
        print(param)
    
    err_correct = param['err_correct']
    smoothed_IGM = param['smoothed_IGM']
    method_IGM = param['method_IGM']
    Nz_IGM = param['Nz_IGM']
    prob_func_IGM = param['prob_func_IGM']
    omegaSN = param['omegaSN [eV]']
    B_IGM = param['B_IGM [nG]']
    ne_IGM = param['ne_IGM [1/cm3]']
    s_IGM = param['s_IGM [Mpc]']
    ICM_effect = param['ICM_effect']
    smoothed_ICM = param['smoothed_ICM']
    method_ICM = param['method_ICM']
    return_arrays = param['return_arrays']  #????
    prob_func_ICM = param['prob_func_ICM']
    Nr_ICM = param['Nr_ICM']
    los_method = param['los_method']
    los_use_prepared_arrays = param['los_use_prepared_arrays']  #????
    los_Nr = param['los_Nr']
    omega_Xrays = param['omegaX [keV]']*1.e3
    omega_CMB = param['omegaCMB [eV]']
    fixed_Rvir = param['fixed_Rvir']
    L_ICM = param['L_ICM [kpc]']
    mu = param['signal_strength']
    ICM_magnetic_model = param['ICM_magnetic_model']

    if ICM_magnetic_model == 'A':

        r_low = 10.
        B_ref = 25.
        r_ref = 0.
        eta = 0.7

    elif ICM_magnetic_model == 'B':

        r_low = 0.
        B_ref = 7.5
        r_ref = 25.
        eta = 0.5

    elif ICM_magnetic_model == 'C':

        r_low = 0.
        B_ref = 4.7
        r_ref = 0.
        eta = 0.5

    else:
        r_low = param['r_low [kpc]']
        B_ref = param['B_ref [muG]']
        r_ref = param['r_ref [kpc]']
        eta = param['eta']

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

    clusters_kwargs = {'omega_Xrays':omega_Xrays,
                      'omega_CMB':omega_CMB,
                      # IGM
                      's_IGM':s_IGM,
                      'B_IGM':B_IGM,
                      'mg_IGM':m_gamma(ne_IGM),
                      'smoothed_IGM':smoothed_IGM,
                      'method_IGM':method_IGM,
                      'prob_func_IGM':prob_func_IGM,
                      'Nz_IGM':Nz_IGM,
                      # ICM
                      'ICM_effect':ICM_effect,
                      'r_low':r_low,
                      'L':L_ICM,
                      'smoothed_ICM':smoothed_ICM,
                      'method_ICM':method_ICM,
                      #'return_arrays':return_arrays,
                      'prob_func_ICM':prob_func_ICM,
                      'Nr_ICM':Nr_ICM,
                      'los_method':los_method,
                      #'los_use_prepared_arrays':los_use_prepared_arrays,
                      'los_Nr':los_Nr,
                      'mu':mu,
                      'B_ref':B_ref,
                      'r_ref':r_ref,
                      'eta':eta}

    if param['use_SH0ES'] is True:
        sh0es_data = load_sh0es(data_path)
    
    if param['use_Pantheon'] is True:
        pan_data = load_pantheon(data_path)
 
    if param['use_BAO_DR12'] is True:
        bao_dr12_data = load_BAO_DR12(data_path)

    if param['use_BAO_lowz'] is True:
        bao_lowz_data = load_BAO_lowz(data_path)        

    if param['use_clusters'] is True:
        clusters_data = load_clusters(data_path)

    total_chi_square = total_chi2(pars, key=var,
                           use_SH0ES=param['use_SH0ES'], sh0es_data=sh0es_data,
                           use_BAO_DR12=param['use_BAO_DR12'], bao_dr12_data=bao_dr12_data,
                           use_BAO_lowz=param['use_BAO_lowz'], bao_lowz_data=bao_lowz_data,
                           use_Pantheon=param['use_Pantheon'], pan_data=pan_data, pan_kwargs=pan_kwargs,
                           use_TDCOSMO=param['use_TDCOSMO'],
                           use_early=param['use_early'], 
                           use_clusters=param['use_clusters'], clusters_data=clusters_data, 
                           err_correct=err_correct, fixed_Rvir=fixed_Rvir, clusters_kwargs=clusters_kwargs,
                           verbose=param['verbose']) 
                         
    return total_chi_square


print(chi_square((1,2,3,4,5,6)))


