#######################################################
###        Likelihood-Ratio Test for ma-ga          ###
###    for a trial project with Prof. Jiji Fan      ###
###              by Cooper Niu, 2022                ###
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
from analysis_main import pltpath


##########################
# auxiliary functions
##########################

def run_lkl_ratio(nbin = 50, pdir = '', ndir =''):
    """
    output the contours in ma-ga space from the frequentist likelihood ratio test.
    """

    #--------------------------#
    #      Read Chain file     #
    #--------------------------#

    flgn = True

    directory = pdir
    neg_dir = ndir
    bins = nbin

    # reading chains
    path = directory+'/chain.h5'
    f = h5py.File(path, 'r')

    f = f['mcmc']
    keys = f.keys()

    pts = np.array(f['chain']) # the points
    pts = pts.reshape(-1, 6)
    
    chi2_tot = np.array(f['log_prob'])
    chi2_tot *= -2
    chi2_tot = chi2_tot.reshape(-1)

    blobs = f['blobs']
    experiments = dict(blobs.dtype.fields).keys()

    each_chi2 = {exper:blobs[exper].reshape(-1) for exper in experiments} 

    del f
    
    bf_chi2, bf_idx = min(chi2_tot), chi2_tot.argmin() 
    each_sum = sum([each_chi2[exper][bf_idx] for exper in experiments]) 
#    print "chi2 best fit: {} = {}".format(bf_chi2, each_sum) # sanity check
    
    if flgn:
        
        neg_path = neg_dir+'/chain.h5'
        nf = h5py.File(neg_path, 'r')
        
        nf = nf['mcmc']
        nkeys = nf.keys()
        
        npts = np.array(nf['chain'])
        npts = npts.reshape(-1, 6)

        nchi2_tot = np.array(nf['log_prob'])
        nchi2_tot *= -2
        nchi2_tot = nchi2_tot.reshape(-1)
        
        nblobs = nf['blobs']
        nexperiments = dict(nblobs.dtype.fields).keys()
        
        neach_chi2 = {nexper:nblobs[nexper].reshape(-1) for nexper in nexperiments} 
        
        del nf
        
        nbf_chi2, nbf_idx = min(nchi2_tot), nchi2_tot.argmin() 
        neach_sum = sum([neach_chi2[nexper][nbf_idx] for nexper in nexperiments])         
#        print "chi2 best fit for negative signal strength: {} = {}".format(nbf_chi2, neach_sum) # sanity check

    #-----------------#
    #   print info    #
    #-----------------#

    p_file = '[chi best fit]: '+ str(bf_chi2) + '(p)'+ str(nbf_chi2) + '(n)'+'.txt'
    p_file_path = os.path.join(directory, p_file)
    with open(p_file_path, 'w') as file:
        file.write('nothing here')    
    
    
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
    mesh_ga, mesh_ma = np.meshgrid(block_ga, block_ma, indexing='ij')
    
    
    #--------------------------#
    #     Find chi2(ma, ga)    #
    #--------------------------#
    
    chi2_mins_2D = [] 
    idx_mins_2D = []
    ma_ga_chi2 = [] 
    wheres_2D = {} 
    
    for i in tqdm(range(len(edges_ga)-1)):
        for j in range(len(edges_ma)-1):
            wheres_2D[i,j] = np.where((chain_ga>edges_ga[i])
                                      & (chain_ga<edges_ga[i+1])
                                      & (chain_ma>edges_ma[j])
                                      & (chain_ma<edges_ma[j+1]))

            chi2_ij_block =  chi2_tot[wheres_2D[i,j]]

            if len(chi2_ij_block) > 0:

                min_chi2_ij = min(chi2_ij_block)
                chi2_mins_2D.append(min_chi2_ij)

                idx_mins_2D.append(chi2_ij_block.argmin())
                ma_ga_chi2.append([mesh_ma[i,j], mesh_ga[i,j], min_chi2_ij])

            else:
                chi2_mins_2D.append(np.inf)
                idx_mins_2D.append(-1)

                continue

    chi2_mins_2D = np.array(chi2_mins_2D)
    idx_mins_2D = np.array(idx_mins_2D, dtype=int)

    chi2_mins_2D = chi2_mins_2D.reshape(mesh_ma.shape)
    idx_mins_2D = idx_mins_2D.reshape(mesh_ma.shape)

    ma_ga_chi2 = np.array(ma_ga_chi2)
    chi2_ma_ga_fn = lndi(ma_ga_chi2[:,0:2], ma_ga_chi2[:,2])
    

    
    #--------------------------#
    #       Find chi2(ma)      #
    #--------------------------#
    
    chi2_mins_1D = [] 
    idx_mins_1D = [] 
    ma_chi2 = []  
    
    for i in tqdm(range(len(edges_ma)-1)):
        
        where = np.where((chain_ma>edges_ma[i])
                         & (chain_ma<edges_ma[i+1]))
        
        chi2_i_block =  chi2_tot[where]
        
        if len(chi2_i_block)>0:
            
            min_chi2_i = min(chi2_i_block) 
            
            chi2_mins_1D.append(min_chi2_i)
            idx_mins_1D.append(chi2_i_block.argmin())
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
    
    if flgn:
        
        nchain_ma = npts[:,2] 
        nchi2_mins_1D = [] 
        nidx_mins_1D = []
        nma_chi2 = [] 
        
        for i in tqdm(range(len(edges_ma)-1)):
            
            where = np.where((nchain_ma>edges_ma[i])
                             & (nchain_ma<edges_ma[i+1]))
            
            nchi2_i_block =  nchi2_tot[where]
            
            if len(nchi2_i_block)>0:
                
                nmin_chi2_i = min(nchi2_i_block) 
                
                nchi2_mins_1D.append(nmin_chi2_i)
                nidx_mins_1D.append(nchi2_i_block.argmin())
                nma_chi2.append([block_ma[i], nmin_chi2_i])
            
            else:
                nchi2_mins_1D.append(np.inf)
                nidx_mins_1D.append(-1)
                
                continue
        
        nchi2_mins_1D = np.array(nchi2_mins_1D)
        nidx_mins_1D = np.array(nidx_mins_1D, dtype=int)
        
        nchi2_mins_1D = nchi2_mins_1D.reshape(block_ma.shape)
        nidx_mins_1D = nidx_mins_1D.reshape(block_ma.shape)
        
        nma_chi2 = np.array(nma_chi2)
        nchi2_ma_fn = interp1d(nma_chi2[:,0], nma_chi2[:,-1], fill_value="extrapolate")
    
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
    
    # defining the fixed-ma best-fit chi2
    if not flgn:
        bf_chi2_ma = chi2_ma_fn(ma_gr)
        neg_str = ""
    else:
        bf_chi2_ma = np.minimum.reduce([chi2_ma_fn(ma_gr), nchi2_ma_fn(ma_gr)])
        neg_str = "_neg"
    
    # array of delta chi2
    delta_arr = chi2_ma_ga_fn(ma_gr, ga_gr) - bf_chi2_ma
     
    # the points of the 2-sigma (95% C.L.) contour for a one-sided test (2.705543 chi2 threshold)
    cs = plt.contour(ma_arr, ga_arr, delta_arr, levels=[2.705543])
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    np.savetxt(pltpath(directory, head='one-sided_95CL'+neg_str, ext='.txt'), v)
    
    # the delta_chi2 contour
    plt.contour(ma_arr, ga_arr, delta_arr, levels=[2.705543], colors=['blue'], linestyles=['-'], )
    
    if flgn: 
        plt.contour(ma_arr, ga_arr, (chi2_ma_ga_fn(ma_gr, ga_gr) - chi2_ma_fn(ma_gr)), levels=[2.705543], colors=['red'], linestyles=['--'])
    
    plt.savefig(pltpath(directory, head='delta_chi2'+neg_str))
    plt.clf()


if __name__ == '__main__':
    warnings.filterwarnings('error', 'overflow encountered')
    warnings.filterwarnings('error', 'invalid value encountered')
    warnings.filterwarnings('ignore')
 
    argv = sys.argv[1:]

    help_msg = 'Usage: python main.py [option] ... [arg] ...'

    try:
        opts, args = getopt.getopt(argv, 'hp:n:b:')
    except getopt.GetoptError:
        raise Exception(help_msg)

    for opt, arg in opts:
        if opt == '-h':
            print(help_msg)
            sys.exit()
        elif opt == '-n':
            ndir = arg
            flg_n = True
        elif opt == '-p':
            pdir = arg
            flg_w = True
        elif opt == '-b':
            nbins = int(arg)
            flg_b = True

    run_lkl_ratio(nbin = 50, pdir = str(pdir), ndir =str(ndir))




