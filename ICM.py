###############################################################
###                Intracluster Medium Code                 ###
###         for a trial project with Prof. Jiji Fan         ###
###                   by Cooper Niu, 2022                   ###
###############################################################


#=================#
# Import Packages #
#=================#

from __future__ import division
import numpy as np
from scipy.integrate import simps, quad
from numpy import pi, sqrt, log, log10, exp, power, abs
from ag_conversion import P_ag, m_gamma
from inspect import getargspec

#=============================#
# Initialization of Constants #
#=============================#

c0 = 299792458.                 # speed of light [m/s]
alpha = 1./137                  # fine-struction constant
me = (0.51099895 * 1.e6)        # [eV] electron mass
m_Mpc = 3.085677581282e22       # [m/Mpc]
hbarc = 1.9732698045930252e-16  # Length Conversion from GeV^-1 to m [GeV*m]
eV_kg = 5.6095886e35            # Mass Conversion from kg to eV [eV/kg]
eV2_G = 1.95e-2                 # Magnetic Field Strength Conversion from Gauss to eV^2 [eV^2/G]

#===========#
# Functions #
#===========#

# Power law distribution of magnetic domain sizes
def L_size(L):
    """
    L : domain size [kpc]
    """
  
    # Here we set the domain size lower bound and upper bound to be 3.5 and 10. respectively, and the power is -1.2
    L_size = L**-1.2 / ((10.**-0.2/-0.2) - (3.5**-0.2/-0.2))
    return L_size

L_ICM = quad(lambda x: L_size(x) * x, 3.5, 10.)[0]
"""
L_ICM = 6.08032 kpc to be the uniform size of the magnetic domain
"""


# Electron number density [cm^-3] in the double-beta profile
def ne_2beta(r, ne0=0.01,
             rc_outer=100.,
             beta_outer=1.,
             f_inner=0.,
             rc_inner=10.,
             beta_inner=1.):
    """
    r : distance from the center of the cluster [kpc]
    ne0 : central electron number density [cm^-3]
    rc_outer : core radius from the outer component [kpc] (default: 100.)
    beta_outer : slope from the outer component (default: 1.)
    f_inner : fractional contribution from inner component (default: 0.)
    rc_inner : core radius from the inner component [kpc] (default: 10.)
    beta_inner : slope from the inner component (default: 1.)
    """

    outer = (1. + r**2. / rc_outer**2.)**(-1.5 * beta_outer) # outer contribution
    inner = (1. + r**2. / rc_inner**2.)**(-1.5 * beta_inner) # inner contribution
    
    return ne0 * (f_inner * inner + (1. - f_inner) * outer)


# Magnetic field in the ICM [muG]
def B_icm(r, B_ref=10.,
          r_ref=0.,
          eta=0.5, 
 
          # ne_2beta parameters
          ne0=0.01,
          rc_outer=100.,
          beta_outer=1.,
          f_inner=0.,
          rc_inner=10.,
          beta_inner=1.):
    """
    r : distance from the center of the cluster [kpc]
    B_ref : reference value of the magnetic field [muG] (default: 10.)
    r_ref : reference value of the radius [kpc] (default: 0.)
    eta : power of B_icm (default: 0.5)
    others : ne_2beta function parameters
    """
    
    ne_r = ne_2beta(r, ne0 = ne0, 
                    rc_outer = rc_outer,
                    beta_outer = beta_outer,
                    f_inner = f_inner,
                    rc_inner = rc_inner,
                    beta_inner = beta_inner)
 
    ne_ref = ne_2beta(r_ref, ne0 = ne0, 
                      rc_outer = rc_outer, 
                      beta_outer = beta_outer,
                      f_inner = f_inner,
                      rc_inner = rc_inner,
                      beta_inner = beta_inner)

    return B_ref * (ne_r / ne_ref)**eta

# print(B_icm(1, r_ref = 10))
# print(getargspec(ne_2beta))
# print(getargspec(B_icm))
  
# ICM survival probability for photons
def icm_Psurv(ma, g, r_ini, r_fin,
              L=10.,
              omega_Xrays=10.,
              axion_ini_frac=0.,
              smoothed=False,
              method='product',
              prob_func='norm_log',
              Nr=501,
              mu=1.,
              
              # B_icm parameters
              B_ref=10.,
              r_ref=0.,
              eta=0.5,        
              
              # ne_2beta parameters
              ne0=0.01,
              rc_outer=100.,
              beta_outer=1.,
              f_inner=0.,
              rc_inner=10.,
              beta_inner=1.):
    """
    ma : axion mass [eV]
    g : axion-photon coupling [GeV^-2]
    r_ini : photon initial radial distance to the cluster center [kpc]
    r_fin : photon final radial distance to the cluster center [kpc]
    L : ICM magnetic field domain size [kpc] (default: 10.)
    omega_Xrays : photon energy [keV] (default: 10.)
    axion_ini_frac : the initial intensity fraction of axions: I_axion/I_photon (default: 0.)
    smoothed : whether sin^2 in conversion probability is smoothed out [bool] (default: False)
    method : the integration method 'simps'/'quad'/'product' (default: 'product')
    prob_func : the form of the probability function: 'small_P'/'full_log'/'norm_log' [str] (default: 'norm_log')
    Nr : number of radius bins for Simpson integral (default: 501)
    mu : signal strength (default: 1.)
    others : 'ne_2beta' and 'B_icm' parameters
    """
    
    A = (2./3)*(1 + axion_ini_frac)
    
    # ICM electron number density [cm^-3]
    ne = lambda x: ne_2beta(x, ne0 = ne0,
                            rc_outer = rc_outer,
                            beta_outer = beta_outer,
                            f_inner = f_inner,
                            rc_inner = rc_inner,
                            beta_inner = beta_inner)
 			
    # photon plasma mass [eV]
    mg = lambda x: m_gamma(ne(x))

    # ICM magnetic field [muG]
    B_field = lambda x: B_icm(x, B_ref = B_ref,
                              r_ref = r_ref,
                              eta = eta, 
							  ne0 = ne0,
                              rc_outer = rc_outer,
                              beta_outer = beta_outer,
                              f_inner = f_inner,
                              rc_inner = rc_inner,
                              beta_inner = beta_inner)


    # conversion probability in domain located at radius x from center of cluster    
    P = lambda x: mu * P_ag(ma, g, 
                            L/1000., 
                            B = B_field(x) * 1000., 
                            omega = omega_Xrays * 1000., 
                            mg = mg(x), 
                            smoothed = smoothed) 

    ### Integral ###
    
    if method == 'product':

        N = int(round((r_fin - r_ini)/L)) # number of magnetic domains
        r_Arr = (r_ini + L / 2.) + L*np.arange(N) # array of r-values of the domains' centers
        P_Arr = P(r_Arr) # array of conversion probabilities
        factors = 1. - 1.5 * P_Arr # the factors in each domain

        total_prod = factors.prod()
        P_survival = A + (1. - A) * total_prod
 
        return P_survival

    elif method == 'simps':

        rArr = np.linspace(r_ini, r_fin, Nr)

        if prob_func == 'norm_log':
            integrand = log(abs(1. - 1.5*P(rArr)) )
        elif prob_func == 'small_P':
            integrand = -1.5*P(rArr)
        elif prob_func == 'full_log':
            integrand = log( 1. - 1.5*P(rArr) )
        else:
            raise ValueError("Log Method Error!")

        integral = simps(integrand, rArr)
        P_survival = A + (1. - A)*exp(integral / L)

        return P_survival

    elif method == 'quad':

        if prob_func == 'norm_log':
            integrand = lambda x: log(abs(1. - 1.5 * P(x)))
        elif prob_func == 'small_P':
            integrand = lambda x: - 1.5 * P(x)
        elif prob_func == 'full_log':
            integrand = lambda x: log(1. - 1.5 * P(x))
        else:
            raise ValueError("Log Method Error!")
        
        integral = quad(integrand, r_ini, r_fin)[0]
        P_survival = 1. - (1. - A)*(1. - exp(integral / L)) 

        return P_survival

    else:
        raise ValueError("Integral Method Error!")

# Line-of-sight average of the photons ICM survival probability.
def icm_los_Psurv(ma, g, r_low, r_up, 
                  L=10.,
                  omega_Xrays=10.,
                  axion_ini_frac=0.,
                  smoothed=False,
                  method='product',
                  prob_func='norm_log',
                  Nr=501,
                  los_method='quad',
                  los_Nr=501,
                  mu = 1.,

                  # B_icm parameters
                  B_ref=10.,
                  r_ref=0.,
                  eta=0.5,
  
                  # ne_2beta parameters
                  ne0=0.01,
                  rc_outer=100.,
                  beta_outer=1.,
                  f_inner=0.,
                  rc_inner=10.,
                  beta_inner=1.):
    """
    ma : axion mass [eV]
    g : axion-photon coupling [GeV^-2]
    r_low : lower end of the integration [kpc]
    r_up : upper end of the integration [kpc]
    L : ICM magnetic field domain size [kpc] (default: 10.)
    omega_Xrays : photon energy [keV] (default: 10.)
    axion_ini_frac : the initial intensity fraction of axions: I_axion/I_photon (default: 0.)
    smoothed : whether sin^2 in conversion probability is smoothed out [bool] (default: False)
    method : the integration method 'simps'/'quad'/'product' (default: 'product')
    prob_func : the form of the probability function: 'small_P' for the P<<1 limit, 'full_log' for log(1-1.5*P), and 'norm_log' for the normalized log: log(abs(1-1.5*P)) [str] (default: 'norm_log')
    Nr : number of radius bins, for the 'simps' methods (default: 501)
    los_method : the integration method along the line of sight 'simps'/'quad' (default: 'simps')
    los_Nr : number of radius bins along the line of sight, for the 'simps' methods (default: 501)
    mu : signal strength (default: 1.)
    others : 'ne_2beta' and 'B_icm' parameters
    """
    
    # ICM electron number density [cm^-3]
    ne = lambda x: ne_2beta(x, ne0 = ne0,
                            rc_outer = rc_outer,
                            beta_outer = beta_outer,
                            f_inner = f_inner,
                            rc_inner = rc_inner,
                            beta_inner = beta_inner)

    P_gg_ne2 = lambda x: ne(x)**2, * P_icm(ma, g, x, r_up, 
                                           L=L,
                                           omega_photon = omega_photon,
                                           axion_ini_frac = axion_ini_frac,
                                           smoothed = smoothed,
                                           method = method,
                                           prob_func = prob_func,
                                           Nr = Nr,
                                           mu = mu,
						   				   B_ref = B_ref,
										   r_ref = r_ref,
										   eta = eta,
		                                   ne0 = ne0,
										   rc_outer = rc_outer,
                                           beta_outer = beta_outer,
                                           f_inner = f_inner,
                                           rc_inner = rc_inner,
                                           beta_inner = beta_inner)


    if los_method == 'quad': # this method requires functions

        num = quad(P_gg_ne2, r_low, r_up)[0]
        den = quad(ne**2., r_low, r_up)[0]

    elif los_method == 'simps': # this method requires arrays
        raise ValueError("Simpson Method is reqired!")
    
    # X-ray brightness due to the ICM effect
    S_x = num/den

    return S_x
