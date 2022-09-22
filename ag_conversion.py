##################################################################
###	   		 axion-photon conversion probabilities 		       ###
###	        for a trial project with Prof. Jiji Fan 	       ###
###                      by Cooper Niu, 2022			       ###
##################################################################


#=================#
# Import Packages #
#=================#

from __future__ import division
import numpy as np
from numpy import abs, sin, pi, sqrt, log, log10, exp, power    
import scipy as sp

#=============================#
# Initialization of Constants #
#=============================#

c0 = 299792458.					# speed of light [m/s]
alpha = 1./137 					# fine-struction constant
me = (0.51099895 * 1.e6) 		# [eV] electron mass

#==================#
# Unit Conversions #
#==================#

cm_m = 1.e2						# [cm/m]
eV_GeV = 1.e9					# [eV/GeV]
m_Mpc = 3.085677581282e22		# [m/Mpc]
hbarc = 1.9732698045930252e-16	# Length Conversion from GeV^-1 to m [GeV*m]
eV_kg = 5.6095886e35 			# Mass Conversion from kg to eV [eV/kg]
eV2_G = 1.95e-2                 # Magnetic Field Strength Conversion from Gauss to eV^2 [eV^2/G]
G_nG = 1.e-9					# [G/nG]

#===========#
# Functions #
#===========#

# Effective Photon Mass 
def m_gamma(ne):
    """
    ne: electron number density [cm^-3]    
    """

    m_gamma = 4 * pi * alpha * ne * (hbarc * 1.e11)**3 / me
    
    return sqrt(m_gamma) #[eV]

# Oscillation Wavenumber
def k(ma, g, B=1., omega=1., mg=3.e-15):
    """
    g: axion-photon coupling [GeV^-1]
    ma : axion mass [eV]
    B: magnetic field [nG] (default: 1.)
    omega : photon energy [eV] (default: 1.)
    mg : photon mass [eV] (default: 3.e-15)
    """

    d_a_g = g * 1.e-9 * B * 1.e-9 * 1.95e-2 # Delta_a_gamma [eV]
    d_a = ma**2. / (2. * omega)					# Delta_a [eV]
    d_g = mg**2. / (2. * omega)					# Delta_gamma [eV]

    k = sqrt(d_a_g**2. + (d_a - d_g)**2.)		# wavenumber 
	
    return k #[eV]

# Axion-photon Conversion Probability
def P_ag(ma, g, x, B=1., omega=1., mg=3.e-15, smoothed=False):
    """
    ma : axion mass [eV]
    g : axion-photon coupling [GeV^-1]
    x : distance traveled [Mpc]
    B : magnetic field [nG] (default: 1.)
    omega : photon energy [eV] (default: 1.)
    mg : photon mass [eV] (default: 3.e-15)
    smoothed : whether the sin^2(kx/2) oscillate rapidly within a single domain [bool] (default: False)
    """

    d_a_g = g * 1.e-9 * B * 1.e-9 * 1.95e-2 # Delta_a_gamma [eV]
    coeff = d_a_g**2. / (k(ma, g, B=B, omega=omega, mg=mg)**2.)
    arg = (k(ma, g, B=B, omega=omega, mg=mg) * x * m_Mpc / hbarc * 1.e-9)/2.

    if not smoothed: # normal sine
        osc = np.sin(arg)**2.
    else: # if the sine wave oscillates rapidly, the sin^2(kx/2) can be averaged to 1/2.
        osc = (1 - exp(-2*arg**2.))/2.

#    print("coeff * osc: ", coeff * osc)    

    return coeff * osc

#print(P_ag(1.e-10,2.e-10,1,B=1., omega=1., mg=3.e-15, smoothed=False), P_ag(1.e-10,2.e-10,1,B=1., omega=1., mg=3.e-15, smoothed=True))

def P_survival(ma, g, y, s=1., B=1., omega=1., mg=3.e-15, axion_ini_frac=0., smoothed=False):
    """
    ma : axion mass [eV]
    g : axion-photon coupling [GeV^-1]
    y : comoving distance traveled by the photons [Mpc]
    s : magnetic domain size [Mpc] (default: 1.)
    B : magnetic field [nG] (default: 1.)
    omega : photon energy [eV] (default: 1.)
    mg : photon mass [eV] (default: 3.e-15)
    smoothed : whether the sin^2(kx/2) oscillate rapidly within a single domain [bool] (default: False)
    """

    A = (2./3) * (1 + axion_ini_frac)
    N = y/s
    P = P_ag(ma, g, s, B=B, omega=omega, mg=mg, smoothed = smoothed)
    argument = -1.5 * N * P

    return A + (1-A) * exp(argument)
