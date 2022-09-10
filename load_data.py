##############################################################
###              Loading Experimental Data Code             ###
###         for a trial project with Prof. Jiji Fan         ###
###                   by Cooper Niu, 2022                   ###
###############################################################


#=================#
# Import Packages #
#=================#

import os
from random import sample
import numpy as np
import scipy.linalg as la
from numpy import pi, sqrt, log, log10, exp, power
import pandas as pd
import numexpr as ne
from collections import OrderedDict as od

#===========#
# Functions #
#===========#

c0 = 299792458.                     # speed of light [m/s]
alpha = 1./137                      # fine-struction constant
me = (0.51099895 * 1.e6)            # [eV] electron mass
 
#==================#
# Unit Conversions #
#==================#
 
cm_m = 1.e2                         # [cm/m]
eV_GeV = 1.e9                       # [eV/GeV]
m_Mpc = 3.085677581282e22           # [m/Mpc]
hbarc = 1.9732698045930252e-16      # Length Conversion from GeV^-1 to m [GeV*m]
eV_kg = 5.6095886e35                # Mass Conversion from kg to eV [eV/kg]
eV2_G = 1.95e-2                     # Magnetic Field Strength Conversion from Gauss to eV^2 [eV^2/G]
G_nG = 1.e-9                        # [G/nG]
arcsec_rad = (2.*pi)/(360.*60.*60.) # [arcsec/rad] 


#===================#
# Loading Functions #
#===================#


# Load BAO DR12 Datasets
def load_BAO_DR12(dir, DR12_rsfid = 147.78, DR12_measure = "BAO_result.txt", DR12_covmat = "BAO_covtot.txt"):
    """
    return: BAO_rsfid, BAO_z, BAO_dM, BAO_Hz, BAO_cov, BAO_inv_cov
    """
    
    # Initialize the np array
    BAO_rsfid = DR12_rsfid
    BAO_z = np.array([], 'float64')
    BAO_dM = np.array([], 'float64')
    BAO_Hz = np.array([], 'float64')

    # define the path to files
    result_path = (os.path.join(dir, DR12_measure))
    covmat_path = (os.path.join(dir, DR12_covmat))

    # read the file and append data in the corresponding array
    with open(result_path, 'r') as f:
        for line in f:
            words = line.split()

            if words[0] != '#':
                if words[1] == 'dM(rsfid/rs)':
                    BAO_z = np.append(BAO_z, float(words[0]))
                    BAO_dM = np.append(BAO_dM, float(words[2]))
                elif words[1] == 'Hz(rs/rsfid)':
                    BAO_Hz = np.append(BAO_Hz, float(words[2]))
    
    # Calculate the covariance matrix
    BAO_cov = np.loadtxt(covmat_path)
    BAO_inv_cov = np.linalg.inv(BAO_cov)

    return (BAO_rsfid, BAO_z, BAO_dM, BAO_Hz, BAO_cov, BAO_inv_cov)

# load_BAO_DR12("/Users/cooper/Axion/TProject/datasets")





# Load BAOlowz (6DFs + DR7 MGS)
def load_BAO_lowz(dir, BAO_lowz = "BAO_lowz.txt"):
    """
    return: name, BAO_z, BAO_rs_dV, BAO_sigma, BAO_type
    """
    
    # Initialize the np array
    name = np.array([])
    BAO_z = np.array([], 'float64')
    BAO_rs_dV = np.array([], 'float64')
    BAO_sigma = np.array([], 'float64')
    BAO_type = np.array([], 'int')

    # define the path to files    
    path = (os.path.join(dir, BAO_lowz))
        
    # read the file and append data in the corresponding array
    with open(path, 'r') as f:
        for line in f:
            words = line.split()
            
            if line[0] != '#':
                name = np.append(name, words[0])
                BAO_z = np.append(BAO_z, float(words[1]))
                BAO_rs_dV = np.append(BAO_rs_dV, float(words[2]))
                BAO_sigma = np.append(BAO_sigma, float(words[3]))
                BAO_type = np.append(BAO_type, int(words[4]))
    
    return (name, BAO_z, BAO_rs_dV, BAO_sigma, BAO_type)

# print(load_BAO_lowz("/Users/cooper/Axion/TProject/datasets")[0])



# Load SH0ES dataset
def load_sh0es(dir, SH0ES_data = "SH0ES_data.txt", aB = 0.71273, aBsig = 0.00176):
    """
    return: name, m_SN, SN_sigma, Cepheid, Cepheid_sigma, M, M_sigma
    """
    
    # Initialize the np array
    name = np.array([])
    SN = np.array([], 'float64')
    SN_sigma = np.array([], 'float64')
    Cepheid = np.array([], 'float64')
    Cepheid_sigma = np.array([], 'float64')
    M = np.array([], 'float64')
    M_sigma = np.array([], 'float64')

    # define the path to files
    path = os.path.join(dir, SH0ES_data)

    # read the file and append data in the corresponding array
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if line.strip() and line.find('#') == -1:
                word = line.split()
    
                name = np.append(name, word[0] + ' ' + word[1])
                SN = np.append(SN, float(word[2]))
                SN_sigma = np.append(SN_sigma, float(word[3]))
                Cepheid = np.append(Cepheid, float(word[4]))
                Cepheid_sigma = np.append(Cepheid_sigma, float(word[5]))
                M = np.append(M, float(word[6]))
                M_sigma = np.append(M_sigma, float(word[7]))
    
    # "SN" in the data file plus an extra 5 * aB, we substract it off here
    m_SN = SN - 5 * aB 
    
    return (name, m_SN, SN_sigma, Cepheid, Cepheid_sigma, M, M_sigma)

# print(load_sh0es("/Users/cooper/Axion/TProject/datasets")[0])





# Load Pantheon dataset
def load_pantheon(dir, Pantheon_data = "Pantheon_SN.txt", Pantheon_covmat = "Pantheon_Error.txt", verbose = "2"):
    """
    return: name, pan_z, pan_zhel, pan_dz, pan_dm, pan_dmb, pan_cov
    """

    # Initialize the np array
    data_path = os.path.join(dir, Pantheon_data)
    err_path = os.path.join(dir, Pantheon_covmat)

    # data    
    name = np.array([])
    pan_z = np.array([], 'float64')
    pan_zhel = np.array([], 'float64')
    pan_dz = np.array([], 'float64')
    pan_dm = np.array([], 'float64')
    pan_dmb = np.array([], 'float64')

    # read the file and append data in the corresponding array 
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if line.strip() and line.find('#') == -1:

                word = line.split()

                name = np.append(name, word[0])
                pan_z = np.append(pan_z, float(word[1]))
                pan_zhel = np.append(pan_zhel,float(word[2]))
                pan_dz = np.append(pan_dz, float(word[3]))
                pan_dm = np.append(pan_dm, float(word[4]))
                pan_dmb = np.append(pan_dmb, float(word[5]))

    # covariant matrix
    with open(err_path, 'r') as f:
        length = int(f.readline())

    err_list = pd.read_table(err_path)
    cov_matrix = np.asmatrix(err_list).reshape((length, length))

    pan_cov = ne.evaluate("cov_matrix")
    pan_cov += np.diag(pan_dmb ** 2)
    pan_cov = la.cholesky(pan_cov, lower=True, overwrite_a=True)
 
    return (name, pan_z, pan_zhel, pan_dz, pan_dm, pan_dmb, pan_cov)
    
# print(load_pantheon("/Users/cooper/Axion/TProject/datasets")[6])





# Load ADD
def load_clusters(dir, cluster_data = "cluster_data.txt"):
    """
    return: name, cls_z, cls_DA, cls_err, cls_asymm, cls_ne0, cls_beta, cls_rc_out, cls_f, cls_rc_in, cls_Rvir
    """
    
    # from Bonamente et al., astro-ph/0512349, Table 3.
    stat = np.array([0.01, 0.15, 0.08, 0.08, 0.01, 0.02])
    sys_p = np.array([0.03, 0.05, 0.075, 0.08])
    sys_n = np.array([0.05, 0.075, 0.08])
    
    # Initialize the np array
    name = np.array([])
    cls_z = np.array([], 'float64')
    cls_DA = np.array([], 'float64')
    cls_p_err = np.array([], 'float64')
    cls_n_err = np.array([], 'float64')
    cls_ne0 = np.array([], 'float64')
    cls_beta = np.array([], 'float64')
    cls_rc_out = np.array([], 'float64')
    cls_f = np.array([], 'float64')
    cls_rc_in = np.array([], 'float64')
    cls_Rvir = np.array([], 'float64')
    
    # define the path to files
    path = os.path.join(dir, cluster_data) 

    # read the file and append data in the corresponding array
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if line.strip() and line.find('#') == -1:
                
                this_line = line.split()
                
                name = np.append(name, this_line[0] + ' ' + this_line[1])
                cls_z = np.append(cls_z, float(this_line[2]))
                cls_DA = np.append(cls_DA, float(this_line[3]))
                cls_p_err= np.append(cls_p_err, float(this_line[4]))
                cls_n_err = np.append(cls_n_err, float(this_line[5]))
                cls_ne0 = np.append(cls_ne0, float(this_line[6]))
                cls_beta = np.append(cls_beta, float(this_line[8]))
                cls_rc_out = np.append(cls_rc_out, float(this_line[10]))
                cls_f = np.append(cls_f, float(this_line[12]))
                cls_rc_in = np.append(cls_rc_in, float(this_line[14]))             
                cls_Rvir = np.append(cls_Rvir, float(this_line[19]))
   
    # Convert the unit from arcsec to kpc
    cls_rc_out = cls_DA * cls_rc_out * 1.e3 * arcsec_rad 
    cls_rc_in = cls_DA * cls_rc_in * 1.e3 * arcsec_rad
    cls_Rvir = cls_DA * cls_Rvir * 1.e3 * arcsec_rad
    
    sig_p = sqrt(cls_DA**2 * ((stat**2.).sum() + sys_p.sum()**2.) + cls_p_err**2.)
    sig_m = sqrt(cls_DA**2 * ((stat**2.).sum() + sys_n.sum()**2.) + cls_n_err**2.)
    
    cls_err = (sig_p + sig_m) / 2.
    cls_asymm = (sig_p - sig_m) / (sig_p + sig_m)
    
    #print(name)
   # print(cls_z)
  #  print(cls_DA)
 #   print(cls_Rvir)

    return (name, cls_z, cls_DA, cls_err, cls_asymm, cls_ne0, cls_beta, cls_rc_out, cls_f, cls_rc_in, cls_Rvir)

# print(load_clusters("/Users/cooper/Axion/TProject/datasets")[3])




def load_param(dir):
    """
    load the parameter inputs from the .param files.
    """

    term = {}
    par_names = []

    with open(dir, 'r') as f:
        for line in f:
            if (line.startswith('#')) or (line.startswith('\n')) or (line.startswith('\r')):
                pass
            else:
                word = line.split("=")
                key = word[0].strip()

                try:
                    term[key] = float(word[1])
                except: 
                    term[key] = (word[1]).strip()

                    if term[key] == "True":
                        term[key] = True

                    elif term[key] == "False":
                        term[key] = False
                    
                    elif term[key][0] == '[' and term[key][-1] == ']':

                        term[key] = eval(term[key])
 
                        term[key+'_mean'] = term[key][0]
                        term[key+'_low'] = term[key][1]
                        term[key+'_up'] = term[key][2]
                        term[key+'_sig'] = term[key][3]
                        par_names.append(str(key))    
                
    return (term, par_names)                


# print(load_param("/Users/cooper/Axion/test/inputs/neg_sig_example.param"))

















