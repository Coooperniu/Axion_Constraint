###############################################################
###                Intergalactic Medium Data                ###
###         for a trial project with Prof. Jiji Fan         ###
###                   by Cooper Niu, 2022                   ###
###############################################################


#=================#
# Import Packages #
#=================#

from __future__ import division 
import numpy as np
from scipy.integrate import simps, quad
import warnings
from numpy import pi, sqrt, log, log10, exp, power
from ag_conversion import P_ag

#=============================#
# Initialization of Constants #
#=============================#

c0 = 299792458.                 # speed of light [m/s]
alpha = 1./137                  # fine-struction constant
me = (0.51099895 * 1.e6)        # [eV] electron mass

#===========#
# Functions #
#===========#

# Hubble expansion rate in flat LambdaCDM
def H(Omega_L, z):
    """
    Omega_L : cosmological constant fractional density
    z : redshift
    """

    H = sqrt(Omega_L + (1 - Omega_L) * (1. + z)**3.)
    return H

#print(H(0.7,1)* (0.7 * 1.e2)/(c0 * 1.e-3))

# Comoving Distance [Mpc]
def Comove_D(z, h = 0.7, Omega_L = 0.7):
    """
    z : redshift
    h : reduced Hubble parameter H0/100 [km/s/Mpc] (default: 0.7)
    Omega_L : cosmological constant fractional density (default: 0.7)
    """
    
    D_H = (c0 * 1.e-3) / (h * 1.e2) # Hubble Distance [Mpc] 

    comoving_distance = D_H * quad(lambda x: 1. / H(Omega_L, x), 0., z)[0]

    return comoving_distance

# print(Comove_D(1))

# Photon IGM survival probability
def P_igm(ma, g, z,
          s=1.,
          B=1.,
          omega=1.,
          mg=3.e-15,
          h=0.7,
          Omega_L=0.7,
          axion_ini_frac=0.,
          smoothed=False,
          method='simps',
          prob_func='norm_log',
          Nz=501,
          mu=1.):
    """
    ma : axion mass [eV]
    g : axion-photon coupling [GeV^-2]
    z : redshift
    s : magnetic domain size [Mpc] (default: 1.)
    B : magnetic field  [nG] (default: 1.)
    omega : photon energy [eV] (default: 1.)
    mg : photon mass [eV] (default: 3.e-15)
    h : reduced Hubble parameter H0/100 [km/s/Mpc] (default: 0.7)
    Omega_L : cosmological constant fractional density (default: 0.7)
    axion_ini_frac : the initial intensity fraction of axions: I_axion/I_photon (default: 0.)
    smoothed : whether the sin^2(kx/2) oscillate rapidly within a single domain [bool] (default: False)
    method : the integration method 'simps'/'quad' (default: 'simps')
    prob_func : the form of the probability function: 'small_P'/'full_log'/'norm_log' [str] (default: 'norm_log')
    Nz : number of redshift bins, for the 'simps' methods (default: 501)
    mu : signal strength (default: 1.)
    """

    A = (2./3)*(1 + axion_ini_frac)
    D_H = (c0 * 1.e-3) / (h * 1.e2) # Hubble Distance [Mpc] 
         
    P_gamma = lambda x: mu * P_ag(ma, g, s/(1+x), 
								  B = B*(1+x)**2., 
								  omega = omega*(1.+x), 
								  mg = mg*(1+x)**1.5, 
								  smoothed=smoothed)
       
    ### integral ###

    argument = 0.

    if method == 'simps':
        
        if z <= 1.e-10:
            z_array = np.linspace(0., 1.e-10, int(Nz))
        else:
            z_array = np.linspace(0., z, int(Nz))

        if prob_func == 'norm_log':
            integrand = log(abs(1 - 1.5*P_gamma(z_array))) / H(Omega_L, z_array)
        elif prob_func == 'small_P':
            integrand = - 1.5 * P_gamma(z_array) / H(Omega_L, z_array)
        elif prob_func == 'full_log':
            integrand = log( 1 - 1.5 * P_gamma(z_array) ) / H(Omega_L, z_array)
        else:
            raise ValueError("Log Method Error!")

        argument = (D_H/s) * simps(integrand, z_array) # argument of the exponential
#        print("argument:", argument) 
              
    elif method == 'quad':

        if prob_func == 'norm_log':
            integrand = lambda x: log(abs(1 - 1.5*P_gamma(x))) / H(Omega_L, x)
        elif prob_func == 'small_P':
            integrand = lambda x: -1.5 * P_gamma(x) / H(Omega_L, x)
        elif prob_func == 'full_log':
            integrand = lambda x: log( 1 - 1.5 * P_gamma(x) ) / H(Omega_L, x)
        else:
            raise ValueError("Log Method Error!")

        argument = (D_H / s) * quad(integrand, 0., z)[0]
  
    else:
        raise ValueError("Function Method Error!")
    
    warnings.filterwarnings('ignore')      
    P_conv = (1. - A) * ( 1. - exp(argument))
    P_survival = 1. - P_conv

#    print("A: ",A)
#    print("exp:",exp(argument))
#    print("P_igm: ", P_survival)
    return P_survival

# Sanity Test
# print(P_igm(1.3214621361,2.3214621361,3.3214621361, method='simps'), P_igm(1.3214621361,2.3214621361,3.3214621361, method='quad'))
