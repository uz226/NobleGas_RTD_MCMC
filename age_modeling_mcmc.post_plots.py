# Modeling age distributions using MCMC analysis
#
# Worflow is:
# 1 - Read in observation datasets, including the modeled He_terrigenic from the ng_interp dir
# 2 - Read in posterior CE model parameter sets from the ng_interp dir
# 3 - Read in tracer atmospheric histories
#
# This script writes some pickles that hold the data necassary to run MCMC
# MCMC is actually performed in age_ens_runs dir



import numpy as np
import pandas as pd
import pickle
import sys
import datetime
import pdb
import os
import time


import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, MaxNLocator, LogLocator)
import matplotlib.ticker as ticker
plt.rcParams['font.size']=14
#plt.rcParams["text.usetex"] = False
import matplotlib.style as style
style.use('tableau-colorblind10')


sys.path.insert(0, './utils')
import noble_gas_utils as ng_utils
import convolution_integral_utils as conv
import cfc_utils as cfc_utils

#import pymc3 as mc
#import theano
#import theano.tensor as TT
#from theano import as_op
#from theano.compile.ops import as_op
import arviz as az

import copy
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# Read in pickles from age_modeling_mcmc.prep.py
map_dict = pd.read_pickle('./map_dict.pk')
ens_dict = pd.read_pickle('./ens_dict.pk')
C_in_dict = pd.read_pickle('./C_in_dict.pk')



# Plotting label utils
ylabs = {'CFC11':'CFC-11',
         'CFC12':'CFC-12',
         'CFC113':'CFC-113',
         'SF6':'SF$_{6}$',
         'He4_ter':'$^{4}$He$\mathrm{_{ter}}$',
         'H3_He3':'$\mathrm{^{3}H/^{3}He}$',
         'H3':'$\mathrm{^{3}H}$',
         'He3':'$\mathrm{^{3}He}$'}

ylabs_units = {'CFC11':'pptv',
               'CFC12':'pptv',
               'CFC113':'pptv',
               'SF6':'pptv',
               'He4_ter':'cc/g',
               'H3_He3':'TU',
               'H3':'TU',
               'He3':'TU'}

var_map = {'tau1':r'$\it{\tau_{1}}}$ (yrs)',
           'tau2':r'$\it{\tau_{2}}$ (yrs)',
           'tau':r'$\it{\tau_{comp}}$ (yrs)',
           'f1':'$\it{f_{1}}$ (-)',
           'f2':r'$\it{f_{2}}$ (-)',
           'eta1':'$\it{\eta_{1}}$ (-)',
           'eta2':'$\it{\eta_{2}}$ (-)',
           'nu':r'$\it{\nu}$ (-)',
           'D1':r'$\it{D_{1}}$ (-)',
           'D2':r'$\it{D_{2}}$ (-)',
           'J':r'$\it{J}$ (cm$^{3}\,g^{-1}\,yr^{-1}$)',
           'thalf_cfc':r'$\it{CFC12 \ t_{1/2}}$ $(yr^{-1})$',
           'lamsf6':r'$\it{SF_{6}}$ rel. error'}





#--------------------------------------------------
#
# Collect tau optimazation results
#
#--------------------------------------------------
# Read in mcmc traces

#-----------------------
# CHANGE ME
#-----------------------
mc_dir = './age_ens_runs_mcmc/conv_traces'
wells     = ['PLM1', 'PLM7', 'PLM6']
#----------------------


def to_hdi(trace_arr):
    cl, ch = az.hdi(trace_arr, 0.94)
    #cl, ch = az.hdi(trace_arr, 0.99)
    mask = np.where((trace_arr >= cl) & (trace_arr <= ch), True, False)
    return trace_arr[mask]


# Moving to dataframes to facilitate joint plots with seaborn..
def read_mcmc(tracers, mod_type, savenum):
    mc_dict = {}
    for w in wells:
        well_i = []
        for t in tracers:
            rtd_df = pd.DataFrame()
            try:
                n    = '{}.{}.{}.{}.netcdf'.format(w,t,mod_type,''.join(str(x) for x in savenum))
                #print ('{}  ...Found...'.format(n))
                dd   = az.from_netcdf(os.path.join(mc_dir, n))
                vrs  = list(az.summary(dd.posterior).index)
                for v in vrs:
                    ddv  = np.asarray(dd['posterior'][v])
                    #ddv_ = to_hdi(tau.ravel())
                    rtd_df['tracer'] = len(ddv.ravel())*[t]
                    rtd_df[v]        = ddv.ravel()
                # combine into single df
                well_i.append(rtd_df)
            except OSError:
                print ('{}    ...Not Found....'.format(n))
                pass
            except AttributeError:
                print ('{}    ...Not Found....'.format(n))
                pass
        mc_dict[w] = {}
        try:
            mc_dict[w] = pd.concat(well_i).reset_index(drop=True)
        except ValueError:
            pass
    return copy.deepcopy(mc_dict)
    

tracers  = ['H3', 'CFC12', 'He4_ter', 'SF6', 
            'CFC12.SF6',
            'CFC12.SF6.H3',
            'CFC12.SF6.H3.He4_ter']

#pfm_mc   = read_mcmc(tracers=tracers, mod_type='piston',        savenum=[0])
emm_mc   = read_mcmc(tracers=tracers, mod_type='exponential',   savenum=[0])
epm_mc_123   = read_mcmc(tracers=tracers, mod_type='exp_pist_flow', savenum=[123])
epm_pfm_123 = read_mcmc(tracers=tracers, mod_type='exp_pist_flow-piston', savenum=[1,2,3])







# Plotting
def plot_age_marginal(df, well, tracers, var, modtype, savename):
    mc_df = copy.deepcopy(df)
    w = well
    
    # single well, three tracers
    map_df = pd.DataFrame()
    fig, ax = plt.subplots(len(tracers),1, figsize=(2.0, len(tracers)+1))
    fig.subplots_adjust(hspace=0.85, top=0.98, bottom=0.2, left=0.15)
    for i,t in zip(range(len(tracers)), tracers):
        if var in ['tau1','tau2','tau']:
            dd = mc_df[w][mc_df[w]['tracer'] == t][var].to_numpy().ravel()
        else:
            dd = mc_df[w][mc_df[w]['tracer'] == t][var].to_numpy().ravel()
        
        # contaminated sample
        if t=='SF6' and w=='PLM6':
            sns.kdeplot(to_hdi(dd), ax=ax[i], color='C{}'.format(i), bw_adjust=0.5, fill='C{}'.format(i), linewidth=0.1, alpha=0.01)
            ax[i].text(0.5, 0.5, 'Contam.', bbox=dict(facecolor='white',alpha=0.25), horizontalalignment='center', verticalalignment='center', transform=ax[i].transAxes)
        else:
            sns.kdeplot(to_hdi(dd), ax=ax[i], color='C{}'.format(i), bw_adjust=0.5, fill='C{}'.format(i), linewidth=1.5)
            #sns.kdeplot(dd, ax=ax[i], color='C{}'.format(i), bw_adjust=0.2, fill='C{}'.format(i), linewidth=1.5)
        
        # maximum a posteriori
        grid, pdf = az.kde(to_hdi(dd), bw_fct=2.0)
        _map = grid[pdf.argmax()]
        map_df.loc[t, well] = _map
        
        # Cleanup
        if t=='SF6' and w=='PLM6':
            ax[i].axvline(_map, linestyle='--', color='C{}'.format(i), alpha=0.01)
        else: 
            ax[i].axvline(_map, linestyle='--', color='C{}'.format(i), alpha=0.8)
        ax[i].set_yticks([])
        ax[i].set_yticklabels([])
        ax[i].set_ylabel(ylabs[t], labelpad=2)
        #pdb.set_trace()
        #ax[i].set_xlim(ax[i].get_xlim()[0]*0.90, ax[i].get_xlim()[1])
        ax[i].xaxis.set_major_locator(MaxNLocator(4)) 
        ax[i].minorticks_on()
        ax[i].tick_params(bottom=True, top=False, which='both', labelrotation=35, pad=0.025)
        #ax[i].tick_params(bottom=True, top=False, which='both', labelrotation=0, pad=0.025)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
    #ax[0].set_title(w.split('_')[0])
    #ax[2].set_xlabel(r'{} ${{\tau}}$ (years)'.format(modtype))
    ax[len(tracers)-1].set_xlabel('{} {}\n({})'.format(modtype, var_map[var], w))
    r'{} ${{\tau}}$ (years)'.format(modtype)
    if savename:
        plt.savefig('./figures/{}.svg'.format(savename), format='svg')
    plt.show()
    
    return map_df




# Inference using each tracer individually
# probably just focus on pfm and emm here
tracers   = ['H3', 'SF6', 'CFC12', 'He4_ter']


plm1_mapdf = plot_age_marginal(emm_mc, 'PLM1', tracers, 'tau1', 'EMM', 'PLM1_EMM_tau_margs')
plm7_mapdf = plot_age_marginal(emm_mc, 'PLM7', tracers, 'tau1', 'EMM', 'PLM7_EMM_tau_margs')
plm6_mapdf = plot_age_marginal(emm_mc, 'PLM6', tracers, 'tau1', 'EMM', 'PLM6_EMM_tau_margs')






# Inference using joint tracers
cm = ['C0','C1','C3']

def plot_age_marginal_joint(df, wells, tracers, var, modtype, savename):
    """Plot all three wells at once.
       Plots only one var at a time though"""
    mc_df = copy.deepcopy(df)

    # single well, three tracers
    map_df = pd.DataFrame()
    
    fig, ax = plt.subplots(len(wells),1, figsize=(2.0, len(wells)+1))
    #fig.subplots_adjust(hspace=0.75, top=0.92, bottom=0.22, left=0.15)
    fig.subplots_adjust(hspace=0.85, top=0.98, bottom=0.22, left=0.15)
    
    #for i in range(len(list(mc_df.keys()))):
    #    w = list(mc_df.keys())[i]
    for i in range(len(wells)):
        w = wells[i]
    
        #pdb.set_trace()
        #dd = mc_df[w][var].to_numpy().ravel()
        dd = mc_df[w][mc_df[w]['tracer'] == tracers[0]][var].to_numpy().ravel()
        
        # contaminated sample
        if 'SF6' in tracers[0].split('.') and w in ['PLM6_2021']:
        #if 'SF6'==tracers[0].split('.') and w=='PLM6':
            pdb.set_trace()
            sns.kdeplot(to_hdi(dd), ax=ax[i], color=cm[i], bw_adjust=0.2, fill=cm[i], linewidth=0.1, alpha=0.01)
            #sns.kdeplot(dd, ax=ax[i], color='C{}'.format(i), bw_adjust=0.2, fill='C{}'.format(i), linewidth=0.1, alpha=0.01)
            ax[i].text(0.5, 0.5, 'Contam.', bbox=dict(facecolor='white',alpha=0.25), horizontalalignment='center', verticalalignment='center', transform=ax[i].transAxes)
        else:
            #sns.kdeplot(dd, ax=ax[i], color='C{}'.format(i), bw_adjust=0.2, fill='C{}'.format(i), linewidth=1.5)
            sns.kdeplot(to_hdi(dd), ax=ax[i], color=cm[i], bw_adjust=0.2, fill=cm[i], linewidth=1.5)
            
        # maximum a posteriori
        grid, pdf = az.kde(to_hdi(dd), bw_fct=2.0)
        _map = grid[pdf.argmax()]
        map_df.loc[var, w] = _map
            
        # Cleanup
        ax[i].axvline(_map, linestyle='--', color='C{}'.format(i), alpha=0.8)
        ax[i].set_yticks([])
        ax[i].set_yticklabels([])
        ax[i].set_ylabel(w.split('_')[0], labelpad=2)
        #pdb.set_trace()
        #ax[i].set_xlim(ax[i].get_xlim()[0]*0.90, ax[i].get_xlim()[1])
        ax[i].xaxis.set_major_locator(MaxNLocator(5)) 
        ax[i].minorticks_on()
        ax[i].tick_params(bottom=True, top=False, which='both', labelrotation=45, pad=0.025)
        #ax[i].tick_params(bottom=True, top=False, which='both', labelrotation=0, pad=0.025)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
    # Make all x-axis the same?
    ax_lo = [ax[z].get_xlim()[0] for z in range(len(wells))]
    ax_hi = [ax[z].get_xlim()[1] for z in range(len(wells))]
    [ax[z].set_xlim(min(ax_lo), max(ax_hi)) for z in range(len(wells))]
    
    ax[len(wells)-1].set_xlabel(r'{} {}'.format(modtype, var_map[var]), labelpad=0.5)
    # Build a title with tracers?
    #tt = [ylabs[i] for i in tracers[0].split('.')]
    #fig.suptitle(','.join(tt), fontsize=12)
    #fig.tight_layout()
    if savename:
        fdir = os.path.join('./figures',tracers[0]+'_post_figs')
        if os.path.exists(fdir) and os.path.isdir(fdir):
            pass
        else:
            os.makedirs(fdir)
        plt.savefig('{}/{}.svg'.format(fdir, savename), format='svg')
    plt.show()
    
    return map_df





tracers   = ['CFC12.SF6.H3.He4_ter']
wells     = ['PLM1','PLM7','PLM6']

# EPM 123
epm_tau = plot_age_marginal_joint(epm_mc_123, wells, tracers, 'tau1', 'EPM', '123_EPM_tau')
epm_eta = plot_age_marginal_joint(epm_mc_123, wells, tracers, 'eta1', 'EPM', '123_EPM_eta')
epm_J   = plot_age_marginal_joint(epm_mc_123, wells, tracers, 'J',        '', '123_EPM_J')
epm_cfc = plot_age_marginal_joint(epm_mc_123, wells, tracers, 'thalf_cfc','', '123_EPM_thalf_cfc')
epm_sf6 = plot_age_marginal_joint(epm_mc_123, wells, tracers, 'lamsf6',   '', '123_EPM_lam_sf6')

# EPM-PFM 123
_tau1  = plot_age_marginal_joint(epm_pfm_123, wells, tracers, 'tau1',     r'EPM$_{{1}}$', '123_EPM_PFM_tau1')
_tau2  = plot_age_marginal_joint(epm_pfm_123, wells, tracers, 'tau2',     r'PFM$_{{2}}$', '123_EPM_PFM_tau2')
_tau   = plot_age_marginal_joint(epm_pfm_123, wells, tracers, 'tau',      '',             '123_EPM_PFM_tau')
_f1    = plot_age_marginal_joint(epm_pfm_123, wells, tracers, 'f1',       r'EPM$_{{1}}$', '123_EPM_PFM_f1')
_eta1  = plot_age_marginal_joint(epm_pfm_123, wells, tracers, 'eta1',     r'EPM$_{{1}}$', '123_EPM_PFM_eta1')
_J     = plot_age_marginal_joint(epm_pfm_123, wells, tracers, 'J',        '',             '123_EPM_PFM_J')
_cfc   = plot_age_marginal_joint(epm_pfm_123, wells, tracers, 'thalf_cfc','',             '123_EPM_PFM_thalf_cfc')
_sf6   = plot_age_marginal_joint(epm_pfm_123, wells, tracers, 'lamsf6',   '',             '123_EPM_PFM_lam_sf6')












#-------------------------------------------------
# Plots of Posterior CDF of young fraction RTD
#-------------------------------------------------

# call convolution integral to evaulate full rtd as function of tau and other parameters


def plot_younger(df, wells, tracers, modtype, savename):
    # Evaulates the weighting function of the convolution integral
    # only works with modtype=='exp_pist_flow' right now
    dd = copy.deepcopy(df)    

    # initialize the convolution intergral with any input, only interested in weigting function here
    tr_in = copy.deepcopy(C_in_dict)['CFC12']
    conv_ = conv.tracer_conv_integral(C_t=tr_in, t_samp=tr_in.index[-1])

    mod_out_dict = {}
    for w in wells:
        mc_df   = copy.deepcopy(dd)
        mod_out = mc_df[w][mc_df[w]['tracer'] == tracers]
        mod_out_dict[w] = mod_out


    # plot of young fraction
    fig, axes = plt.subplots(1,3,figsize=(7,2.5))
    fig.subplots_adjust(left=0.1,right=0.96,bottom=0.24,top=0.90,wspace=0.25)
    gt_cdf_ = []
    for i in range(len(wells)):
        ax = axes[i]
        w  = wells[i]
        for j in range(len(mod_out_dict[wells[0]][::50])):
            try:
                tau1 = mod_out_dict[w]['tau1'].iloc[j]
                eta1 = mod_out_dict[w]['eta1'].iloc[j]
                conv_.update_pars(tau=tau1, eta=eta1, mod_type=modtype)
                gt = conv_.gen_g_tp()[:1000]
                try:
                    gt_cdf = gt.cumsum() / gt.sum()
                    #gt_cdf *= mod_out_dict[w]['f1'].mean()
                    gt_cdf *= copy.deepcopy(mod_out_dict)[w]['f1'].iloc[j]
                except KeyError:
                    gt_cdf = gt.cumsum() / gt.sum()
                    #gt_cdf *= mod_out_dict[w]['f1'].mean()
                    gt_cdf *= 1.0
                # Plot it
                ax.plot(gt_cdf, color='darkgrey', alpha=0.3)
            except IndexError:
                pass
        # MAP prediction
        #tau1_mu = mod_out_dict[w]['tau1'].mean()
        #eta1_mu = mod_out_dict[w]['eta1'].mean()
        tau1_mu = np.median(mod_out_dict[w]['tau1'])
        eta1_mu = np.median(mod_out_dict[w]['eta1'])
        conv_.update_pars(tau=tau1_mu, eta=eta1_mu, mod_type=modtype)
        gt = conv_.gen_g_tp()[:1000]
        gt_cdf = gt.cumsum() / gt.sum()
        try:
            gt_cdf *= copy.deepcopy(mod_out_dict)[w]['f1'].mean()
        except KeyError:
            pass
        gt_cdf_.append(gt_cdf)
        ax.plot(gt_cdf, color='black', linewidth=2.0)
        # Mark the year 1950...
        tt = 2021-1950
        ax.scatter(tt, gt_cdf[tt], marker='o', color='black', zorder=10, facecolors='none', edgecolors='black')
        ax.axhline(gt_cdf[tt], color=cm[i], linestyle='--', linewidth=1.75, alpha=0.75)
        ax.text(tt+4, gt_cdf[tt]-0.07, '1950', va='center', ha='left')
        # Clean-up
        ax.set_xscale('log')
        ax.set_ylim(0.0,1.0)
        ax.yaxis.set_major_locator(MultipleLocator(0.2)) 
        ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        #ax.axhline(0.5, linestyle='--', color='black',alpha=0.5)
        ax.set_xlim(10,1000)
        ax.xaxis.set_major_locator(ticker.LogLocator(numticks=8))
        ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True)
        ax.grid()
        # Label
        ax.text(0.05, 0.9, w.split('_')[0], ha='left', va='center', transform=ax.transAxes)
        
    [axes[i].tick_params(axis='y', labelleft=False) for i in [1,2]]        
    axes[0].set_ylabel('Fraction Younger (-)')
    axes[1].set_xlabel('Residence Times (year)')
    plt.savefig('./figures/tau_young.cdf.{}.{}.png'.format(tracers,savename), dpi=300)
    plt.savefig('./figures/tau_young.cdf.{}.{}.svg'.format(tracers,savename), format='svg')
    plt.show()




wells = ['PLM1','PLM7','PLM6']
tracers = 'CFC12.SF6.H3.He4_ter'


modtype='exp_pist_flow' # only want the young fraction here
plot_younger(df=copy.deepcopy(epm_pfm_123), wells=wells, tracers=tracers, modtype=modtype, savename='epm-pfm.123')


modtype = 'exp_pist_flow'
plot_younger(df=copy.deepcopy(epm_mc_123), wells=wells, tracers=tracers, modtype=modtype, savename='epm.123')










