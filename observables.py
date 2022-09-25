###############################################################
###                Observable Effects Code                  ###
###         for a trial project with Prof. Jiji Fan         ###
###                   by Cooper Niu, 2022                   ###
###############################################################

#=================#
# Import Packages #
#=================#

from __future__ import division
import numpy as np
from numpy import abs, sin, pi, sqrt, log, log10, exp, power, cumprod
import scipy as sp
from scipy.integrate import simps, quad
from scipy.interpolate import interp1d
import warnings

from ag_conversion import m_gamma, k, P_ag, P_survival 
from IGM import H, Comove_D, P_igm
from ICM import ne_2beta, B_icm, P_icm

#=============================#
# Initialization of Constants #
#=============================#

c0 = 299792458.                 # speed of light [m/s]
alpha = 1./137                  # fine-struction constant
me = (0.51099895 * 1.e6)        # [eV] electron mass
 
#==================#
# Unit Conversions #
#==================#
 
cm_m = 1.e2                     # [cm/m]
eV_GeV = 1.e9                   # [eV/GeV]
m_Mpc = 3.085677581282e22       # [m/Mpc]
hbarc = 1.9732698045930252e-16  # Length Conversion from GeV^-1 to m [GeV*m]
eV_kg = 5.6095886e35            # Mass Conversion from kg to eV [eV/kg]
eV2_G = 1.95e-2                 # Magnetic Field Strength Conversion from Gauss to eV^2 [eV^2/G]
G_nG = 1.e-9                    # [G/nG]

#===========#
# Functions #
#===========#


# Angular Diameter Distance (ADD) [Mpc]
def Comove_D(z, h, Omega_L):
    """
    z : redshift
    h : reduced Hubble parameter H0/100 [km/s/Mpc] (default: 0.7)
    Omega_L : cosmological constant fractional density (default: 0.7)
    """

    D_H = (c0 * 1.e-3) / (h * 1.e2) # Hubble Distance [Mpc] 

    comoving_distance = D_H * quad(lambda x: 1. / H(Omega_L, x), 0., z)[0]

    return comoving_distance

# print(Comove_D(1))




def ADD(z, h, Omega_L):
    """
    z : redshift
    h : reduced Hubble parameter H0/100 [km/s/Mpc] (default: 0.7)
    Omega_L : cosmological constant fractional density (default: 0.7) 
    """
    
    D_angular = Comove_D(z, h = h, Omega_L = Omega_L) / (1. + z)
    
    return D_angular

#print(ADD(1))

def Hubble(z, h0, Omega_L, unit='Mpc'):
    if unit == 'Mpc':
        hubble = h0*100.*sqrt(Omega_L + (1 - Omega_L) * (1 + z)**3)/(c0/1000.)
    else:
        hubble = h0*100.*sqrt(Omega_L + (1 - Omega_L) * (1 + z)**3)
    return hubble


def D_Lum(z, h, Omega_L):
    """
    z : redshift
    h : reduced Hubble parameter H0/100 [km/s/Mpc] (default: 0.7)
    Omega_L : cosmological constant fractional density (default: 0.7) 
    """

    D_luminosity = Comove_D(z, h = h, Omega_L = Omega_L) * (1. + z)

    return D_luminosity




# Distance Modulus 'mu' in the standard Lambda-CDM
def mu_mod(z, h, Omega_L):
    """
    z : redshift
    h : reduced Hubble parameter H0/100 [km/s/Mpc] (default: 0.7)
    Omega_L : cosmological constant fractional density 
    """

    std_mu = 25. + 5. * log10(D_Lum(z, h, Omega_L)) 

    return std_mu




# Effective Distand Modulus 'eff_mu'
def eff_mu_mod(ma, g, z,
               s = 1.,
               B = 1.,
               omega = 1.,
               mg = 3.e-15,
               h = 0.7,
               Omega_L = 0.7,
               axion_ini_frac = 0.,
               smoothed = False,
               method = 'simps',
               prob_func = 'norm_log',
               Nz = 501,
               mu = 1.):
    """
    parameters : same as 'P_igm' in 'IGM.py'
    """

    eff_mu = 25. + 5. * log10(D_Lum(z, h, Omega_L)) - 2.5 * log10(P_igm(ma, g, z,
                                                                        s = s,
                                                                        B = B,
                                                                        omega = omega,
                                                                        mg = mg,
                                                                        h = h,
                                                                        Omega_L = Omega_L,
                                                                        axion_ini_frac = axion_ini_frac,
                                                                        smoothed = smoothed,
                                                                        method = method,
                                                                        prob_func = prob_func,
                                                                        Nz = Nz,
                                                                        mu = mu))

    return eff_mu

# The difference between the standard distand modulus and the effective distand modulus (Delta mu)
def delta_mu(ma, g, z,
             s = 1.,
             B = 1.,
             omega = 1.,
             mg = 3.e-15,
             h = 0.7,
             Omega_L = 0.7,
             axion_ini_frac = 0.,
             smoothed = False,
             method = 'simps',
             prob_func = 'norm_log',
             Nz = 501,
             mu = 1.):
    """
    parameters : same as eff_mu_mod
    """

#    delta = mu_mod(z, h = h , Omega_L = Omega_L) - eff_mu_mod(ma, g, z,
#                                                              s = s,
#                                                              B = B,
#                                                              omega = omega,
#                                                              mg = mg,
#                                                              h = h,
#                                                              Omega_L = Omega_L,
#                                                              axion_ini_frac = axion_ini_frac,
#                                                              smoothed = smoothed,
#                                                              method = method,
#                                                              prob_func = prob_func,
#                                                              Nz = Nz,
#                                                              mu = mu)
    
    delta = 2.5 * log10(P_igm(ma, g, z,
                              s = s,
                              B = B,
                              omega = omega,
                              mg = mg,
                              h = h,
                              Omega_L = Omega_L,
                              axion_ini_frac = axion_ini_frac,
                              smoothed = smoothed,
                              method = method,
                              prob_func = prob_func,
                              Nz = Nz,
                              mu = mu))

#    print("delta_mu: ", delta)
    return delta

def P_icm_los(ma, g, r_low, r_up, 
                  L=10.,
                  omega_Xrays=10.,
                  axion_ini_frac=0.,
                  smoothed=False,
                  method='product',
                  prob_func='norm_log',
                  Nr=501,
                  los_method='simps',
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
    prob_func : the form of the probability function: 'small_P'/'full_log'/'norm_log' [str] (default: 'norm_log')
    Nr : number of radius bins, for the 'simps' methods (default: 501)
    los_method : the integration method along the line of sight 'simps'/'quad' (default: 'simps')
    los_Nr : number of radius bins along the line of sight, for the 'simps' methods (default: 501)
    mu : signal strength (default: 1.)
    others : 'ne_2beta' and 'B_icm' parameters
    """
    
    # ICM electron number density [cm^-3]
    ne2 = lambda x: ne_2beta(x, ne0 = ne0,
                            rc_outer = rc_outer,
                            beta_outer = beta_outer,
                            f_inner = f_inner,
                            rc_inner = rc_inner,
                            beta_inner = beta_inner) ** 2

    _, pArr, rArr = P_icm(ma, g, r_low, r_up, 
                          L=L,
                          omega_Xrays = omega_Xrays,
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

    pfn = interp1d(rArr, pArr, fill_value='extrapolate')
    P_gg_ne2 = lambda rr: ne2(rr) * pfn(rr)

    if los_method == 'quad': # this method requires functions

        num = quad(P_gg_ne2, r_low, r_up)[0]
        den = quad(ne2, r_low, r_up)[0]

    elif los_method == 'simps': # this method requires arrays
        
        low_idx = np.abs(rArr - r_low).argmin()
        up_idx = np.abs(rArr - r_up).argmin()
        los_rArr = rArr[low_idx:up_idx+1]
        ne2_Arr = ne2(los_rArr)
        P_gg_ne2_Arr = ne2_Arr * pArr[low_idx:up_idx+1]
   
        del low_idx, up_idx
   
        num = simps(P_gg_ne2_Arr, los_rArr)
        den = simps(ne2_Arr, los_rArr)
        del los_rArr, ne2_Arr, P_gg_ne2_Arr
        
    S_x = num/den
    return S_x






# Effective ADDs
def ADD_mod(ma, g, z, h, Omega_L,

           omega_Xrays=1.e4,
           omega_CMB=2.4e-4,

           # IGM
           s_IGM=1.,
           B_IGM=1.,
           mg_IGM=3.e-15,
           smoothed_IGM=False,
           method_IGM='simps',
           prob_func_IGM='norm_log',
           Nz_IGM=501,

           # ICM
           ICM_effect=False,
           r_low = 0.,
           r_up = 1800.,
           L=10.,
           smoothed_ICM=False,
           method_ICM='product',
           prob_func_ICM='norm_log',
           Nr_ICM=501,
           los_method='quad',
           los_Nr=501,
           mu=1.,

           # B_icm
           B_ref=10.,
           r_ref=0.,
           eta=0.5,

           #ne_2beta
           ne0=0.01,
           rc_outer=100.,
           beta_outer=1.,
           f_inner=0.,
           rc_inner=10.,
           beta_inner=1.):

    if ICM_effect:
    
        prob_gg = P_icm_los(ma, g, r_low, r_up, 
                                   L=L,
                                   omega_Xrays=omega_Xrays/1000.,
                                   axion_ini_frac=0.,
                                   smoothed=smoothed_ICM, 
                                   method=method_ICM, 
                                   prob_func=prob_func_ICM,
                                   Nr=Nr_ICM,
                                   los_method=los_method, 
                                   los_Nr=los_Nr,
                                   mu=mu,

                                   # B_icm
                                   B_ref=B_ref, r_ref=r_ref, eta=eta,

                                   # ne_2beta
                                   ne0=ne0, rc_outer=rc_outer, beta_outer=beta_outer, f_inner=f_inner, rc_inner=rc_inner, beta_inner=beta_inner)

        ini_ag_frac = (1. - prob_gg) / prob_gg
#        print("prob_gg: ", prob_gg)
    else:
        
        prob_gg = 1.
        ini_ag_frac = 0.

    prob_gg_Xray = P_igm(ma, g, z,
                      s=s_IGM,
                      B=B_IGM,
                      omega=omega_Xrays,
                      mg=mg_IGM,
                      h=h,
                      Omega_L=Omega_L,
                      axion_ini_frac=ini_ag_frac,
                      smoothed=smoothed_IGM,
                      method=method_IGM,
                      prob_func=prob_func_IGM,
                      Nz=Nz_IGM,
                      mu=mu)

#    print("prob_gg_Xray: ", prob_gg_Xray)

    prob_gg_CMB = P_igm(ma, g, z,
                        s=s_IGM,
                        B=B_IGM,
                        omega=omega_CMB,
                        mg=mg_IGM,
                        h=h,
                        Omega_L=Omega_L,
                        axion_ini_frac=ini_ag_frac,
                        smoothed=smoothed_IGM,
                        method=method_IGM,
                        prob_func=prob_func_IGM,
                        Nz=Nz_IGM,
                        mu=mu)
    
#    print("prob_gg_CMB: ", prob_gg_CMB)
#    print("....................................")    
    warnings.filterwarnings('ignore')
    # eff_ADD = ADD(z, h = h, Omega_L = Omega_L) * prob_gg_CBM**2 / (prob_gg_Xray * prob_gg)
    eff_ADD = prob_gg_CMB**2 / (prob_gg_Xray * prob_gg) 
    return eff_ADD

#print("standard mu:", mu_mod(0.3,0.7,0.7))
#print("effective mu: " , eff_mu_mod(1.e-10,2.e-5,0.3))
#print("difference: ", mu_mod(0.3,0.7,0.7)- eff_mu_mod(1.e-10,2.e-5,0.3))
#print("delta mu: ", delta_mu(1.e-10,2.e-5,0.3))
#print("ADD_mud: ", ADD_mod(1.e-14,1.e-11,1,0.7,0.7))
#print("++++++++++++++++++++++++++")

