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

# Comoving Distance [Mpc]
def Comove_D(z, h = 0.7, Omega_L)
    """
    z : redshift
    h : reduced Hubble parameter H0/100 [km/s/Mpc] (default: 0.7)
    Omega_L : cosmological constant fractional density (default: 0.7)
    """
    
    D_H = (c0 * 1.e-3) / (h * 1.e2) # Hubble Distance [Mpc] 

    comoving_distance = D_H * quad(lambda z: 1./H(Omega_L, z), 0., z)[0]

    return comoving_distance

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
    smoothed : whether sin^2 in conversion probability is smoothed out [bool] (default: False)
    method : the integration method 'simps'/'quad'/'old' (default: 'simps')
    prob_func : the form of the probability function: 'small_P' for the P<<1, 'full_log' for log(1-1.5*P), and 'norm_log' for the normalized log: log(abs(1-1.5*P)) [str] (default: 'norm_log')
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
        
        if method == 'simps':
        
            if z <= 1.e-10:
                z_array = np.linspace(0., 1.e-10, Nz)
            else:
                z_array = np.linspace(0., z, Nz)

            if prob_func == 'norm_log':
                integrand = log(abs(1 - 1.5*P_gamma(z_array))) / H(Omega_L, z_array)
            elif prob_func == 'small_P':
                integrand = -1.5 * P_gamma(z_array / H(Omega_L, z_array)
            elif prob_func == 'full_log':
                integrand = log( 1 - 1.5 * P_gamma(z_array) ) / H(Omega_L, z_array)
            else:
                raise ValueError("Log Method Error!")

            argument = (D_H/s) * simps(integrand, z_array) # argument of the exponential
        
        elif method == 'quad':

            if prob_func == 'norm_log':
                integrand = lambda x: log(abs(1 - 1.5*Pga(x))) / H(Omega_L, x)
            elif prob_func == 'small_P':
                integrand = lambda x: -1.5 * P_gamma(x) / H(Omega_L, x)
            elif prob_func == 'full_log':
                integrand = lambda x: log( 1 - 1.5 * P_gamma(x) ) / H(Omega_L, x)
            else:
                raise ValueError("Log Method Error!")

            argument = (D_H / s) * quad(integrand, 0., z)[0]
  
        elif method == 'old':
            
            y = comoving_D(z, h=h, Omega_L=Omega_L) 
            argument = -1.5 * (y/s) * P_gamma(z)
            
            #print("the method is old!")

        else:
            raise ValueError("Function Method Error!")
      
    P_survival = 1. - (1.-A)*(1.-exp(argument))

    return P_survival


