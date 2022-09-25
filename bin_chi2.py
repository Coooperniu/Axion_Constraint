#######################################################
###          Code for binned chi2(ma, ga)           ###
###          by Manuel A. Buen-Abad, 2020           ###
###               and Chen Sun, 2020                ###
#######################################################

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

import os
import errno
import sys
import getopt
import warnings
import random
import h5py

import numpy as np
from numpy import pi, sqrt, log, log10, exp, power
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import LinearNDInterpolator as lndi
from tqdm import tqdm
from cosmo_axions_run import pltpath

def run_bin_chi2(input_dir):
    """
    output the contours in ma-ga space from the frequentist likelihood ratio test.
    """

    #--------------------------#
    #      Read Chain file     #
    #--------------------------#

    path = input_dir + 'chain.h5'
    f = h5py.File(path, 'r')

    f = f['mcmc']
    keys = f.keys()
    print(keys)

    pts = np.array(f['chain']) # the points
    pts = pts.reshape(-1, 6)
    
    chi2_tot = np.array(f['log_prob'])
    chi2_tot *= -2
    chi2_tot = chi2_tot.reshape(-1)

    blobs = f['blobs']
    experiments = dict(blobs.dtype.fields).keys()

    each_chi2 = {exper:blobs[exper].reshape(-1) for exper in experiments} # the experiments' chi2s for each point

    del f
    
    bf_chi2, bf_idx = min(chi2_tot), chi2_tot.argmin() # the best fit chi2 and where it is
    each_sum = sum([each_chi2[exper][bf_idx] for exper in experiments]) # the sum of the chi2 from each experiment at the best fit point

    print("chi2 best fit: {} = {}".format(bf_chi2, each_sum)) # sanity check

    #--------------------------#
    #      points and bins     #
    #--------------------------#
    
    # ga:
    chain_ga = pts[:,3] # the values of ga
    chain_neg_ga = chain_ga[np.where(chain_ga<0)] # only negatives!
    _, edges_ga = np.histogram(chain_neg_ga, bins=bins) # the edges of the bins
    block_ga = (edges_ga[:-1] + edges_ga[1:])/2. # center values
    
    # ma:
    chain_ma = pts[:,2] # the values of ma
    chain_neg_ma = chain_ma[np.where(chain_ma<0)] # only negatives!
    _, edges_ma = np.histogram(chain_neg_ma, bins=bins) # the edges of the bins
    block_ma = (edges_ma[:-1] + edges_ma[1:])/2. # center values
    
    # ma-ga:
    # mesh of (ma, ga) parameter space blocks
    mesh_ga, mesh_ma = np.meshgrid(block_ga, block_ma, indexing='ij')

    #--------------------------#
    #     Find chi2(ma, ga)    #
    #--------------------------#

    chi2_mins_2D = [] # the min chi2 in the 2D space
    idx_mins_2D = [] # the index of the min chi2 in the 2D space
    ma_ga_chi2 = [] # the triples (ma, ga, min_chi2) only for those bins where the value is well defined

    wheres_2D = {} # those location indices whose (ma, ga) parameter values are within the bin

    for i in tqdm(range(len(edges_ga)-1)):
        for j in range(len(edges_ma)-1):

            # those points with ga, ma values within the bin (i, j)
            wheres_2D[i,j] = np.where((chain_ga>edges_ga[i])
                                      & (chain_ga<edges_ga[i+1])
                                      & (chain_ma>edges_ma[j])
                                      & (chain_ma<edges_ma[j+1]))

            # the chi2s in the ij-th bin
            chi2_ij_block =  chi2_tot[wheres_2D[i,j]]

            # appending minima and indices
            if len(chi2_ij_block) > 0:

                min_chi2_ij = min(chi2_ij_block) # the minimum chi2 of this bin

                # appending to the list
                chi2_mins_2D.append(min_chi2_ij)
                idx_mins_2D.append(chi2_ij_block.argmin())
                # appending to the data
                ma_ga_chi2.append([mesh_ma[i,j], mesh_ga[i,j], min_chi2_ij])

            else:
                chi2_mins_2D.append(np.inf)
                idx_mins_2D.append(-1)

                continue

    # converting to numpy arrays
    chi2_mins_2D = np.array(chi2_mins_2D)
    idx_mins_2D = np.array(idx_mins_2D, dtype=int)

    chi2_mins_2D = chi2_mins_2D.reshape(mesh_ma.shape)
    idx_mins_2D = idx_mins_2D.reshape(mesh_ma.shape)

    ma_ga_chi2 = np.array(ma_ga_chi2)
    
    # interpolating over the data
    chi2_ma_ga_fn = lndi(ma_ga_chi2[:,0:2], ma_ga_chi2[:,2]) # since data is not a uniform grid, we need to use LinearNDInterpolator


    #--------------------------#
    #       Find chi2(ma)      #
    #--------------------------#

    chi2_mins_1D = [] # the min chi2 in the 1D space
    idx_mins_1D = [] # the index of the min chi2 in the 1D space
    ma_chi2 = [] # the doubles (ma, min_chi2) only for those bins where the value is well defined

    for i in tqdm(range(len(edges_ma)-1)):
        
        # locations in the chain whose ma's are within the i-th ma-bin
        where = np.where((chain_ma>edges_ma[i])
                         & (chain_ma<edges_ma[i+1]))
        
        # the chi2s in that bin
        chi2_i_block =  chi2_tot[where]
        
        # appending minima and indices
        if len(chi2_i_block)>0:
            
            min_chi2_i = min(chi2_i_block) # the minimum chi2 of this bin
            
            # appending to the list
            chi2_mins_1D.append(min_chi2_i)
            idx_mins_1D.append(chi2_i_block.argmin())
            # appending to the data
            ma_chi2.append([block_ma[i], min_chi2_i])
        
        else:
            chi2_mins_1D.append(np.inf)
            idx_mins_1D.append(-1)
            
            continue


    chi2_mins_1D = np.array(chi2_mins_1D)
    idx_mins_1D = np.array(idx_mins_1D, dtype=int)

    chi2_mins_1D = chi2_mins_1D.reshape(block_ma.shape)
    idx_mins_1D = idx_mins_1D.reshape(block_ma.shape)

    ma_chi2 = np.array(ma_chi2)
    chi2_ma_fn = interp1d(ma_chi2[:,0], ma_chi2[:,-1], fill_value="extrapolate")
  

    #------------------------#
    #      Making Plots      #
    #------------------------#
 
    plt.figure(101)
    plt.xlabel(r'$\log_{10} m_a$')
    plt.ylabel(r'$\log_{10} g_a$')
    plt.xlim(-17., -11.)
    plt.ylim(-13., -8.)
    plt.title(r'$\Delta \chi^2$ contours')

    ma_arr = np.linspace(edges_ma[0], edges_ma[-1], 101)
    ga_arr = np.linspace(edges_ga[0], edges_ga[-1], 101)
    ga_gr, ma_gr = np.meshgrid(ga_arr, ma_arr, indexing='ij')

    bf_chi2_ma = chi2_ma_fn(ma_gr)
    neg_str = ""

    delta_arr = chi2_ma_ga_fn(ma_gr, ga_gr) - bf_chi2_ma
    
    # the points of the 2-sigma (95% C.L.) contour for a one-sided test (2.705543 chi2 threshold)
    cs = plt.contour(ma_arr, ga_arr, delta_arr, levels=[2.705543])
    p = cs.collections[0].get_paths()[0]
    v = p.vertices

    np.savetxt(pltpath(directory, head='one-sided_95CL_pts'+neg_str, ext='.txt'), v)
    
    # the delta_chi2 contour
    plt.contour(ma_arr, ga_arr, delta_arr, levels=[2.705543], colors=['blue'], linestyles=['-'])
    
    if flgn: # make the same plot but without the chi2_min from the negative signal strength
        plt.contour(ma_arr, ga_arr, (chi2_ma_ga_fn(ma_gr, ga_gr) - chi2_ma_fn(ma_gr)), levels=[2.705543], colors=['red'], linestyles=['--'])
    
    plt.savefig(pltpath(directory, head='delta_chi2_contour'+neg_str))




 
