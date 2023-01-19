#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 12:42:46 2021

@author: nicholasthiros
"""

# Update 12/10/2021
# Can jointly include multiple tracer concentrations in the likelihood function


# This scripts does not run the MCMC
# Just makes a bunch of posterior plots




import numpy as np
import pandas as pd
import pickle
import sys
import os

import copy

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, MaxNLocator, LogLocator)
import matplotlib.ticker as ticker
plt.rcParams['font.size']=14


import matplotlib.style as style
style.use('tableau-colorblind10')
#style.use('seaborn-colorblind')

import arviz as az

import seaborn as sns


# Where all the magic happens
import run_age_mcmc_utils as age_mcmc

sys.path.insert(0, '../utils')
import noble_gas_utils as ng_utils
import convolution_integral_utils as conv
import cfc_utils as cfc_utils



#------
# Read in obseration ensembles and input histories
map_dict   = pd.read_pickle('./map_dict.pk')
ens_dict   = pd.read_pickle('./ens_dict.pk')
C_in_dict  = pd.read_pickle('./C_in_dict.pk')






ylabs = {'CFC11':'CFC-11',
         'CFC12':'CFC-12',
         'CFC113':'CFC-113',
         'SF6':'SF$_{6}$',
         'He4_ter':'$^{4}$He$\mathrm{_{terr}}$',
         'H3_He3':'$\mathrm{^{3}H/^{3}He}$',
         'H3':'$\mathrm{^{3}H}$',
         'He3':'$\mathrm{^{3}He}$'}

ylabs_units = {'CFC11':'pptv',
               'CFC12':'pptv',
               'CFC113':'pptv',
               'SF6':'pptv',
               'He4_ter':r'10$^{-8}$ cm$^{3}$STP/g',
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
           'J':r'$\it{J}$ (cm$^{3}\,g^{-1}\,yr^{-1}$)',
           'thalf_cfc':r'$\it{CFC-12 \ t_{1/2}}$ $(yr^{-1})$'}




#--------------------------------------------
# Import a traces for multi parameter RTDs
#--------------------------------------------
# Parameters
mc_dir    = './conv_traces'

wells      = ['PLM1', 'PLM7', 'PLM6']


#------
#tracer = 'CFC12'
#tracer = 'H3'
#tracer = 'SF6'
#tracer = 'He4_ter'
tracer   = 'CFC12.SF6.H3.He4_ter'


#----------


#mod_type1  = 'piston'
#mod_type1 = 'exponential'
mod_type1 = 'exp_pist_flow'
#mod_type1 = 'dispersion'


mod_type2 = False
#mod_type2 = 'piston'
#mod_type1 = 'exponential'
#mod_type2 = 'exp_pist_flow'


#---------


#savenum  = [0]    # Constant J, no cfc decay, no sf6 contam.
#savenum  = [1]    # Variable J
#savenum  = [2]    # variable CFC decay
#savenum  = [3]    # variable SF6 contam.

#savenum  = [1,2] # variable J and CFC
savenum  = [1,2,3]
#savenum  = [2,3]


if mod_type2:
    mod_type  = '{}-{}'.format(mod_type1, mod_type2)
else:
    mod_type = mod_type1




#---
# Create a new directory for figures
fdir = './figures/{}.{}.{}.{}.figs'.format('.'.join(wells),mod_type,tracer, ''.join(str(x) for x in savenum))
if os.path.exists(fdir) and os.path.isdir(fdir):
    pass
else:
    os.makedirs(fdir)





#-----
# Read traces
dd_comp = {}
trace_summary_comp = {}

for w in wells:
    try:
        n    = '{}.{}.{}.{}.netcdf'.format(w,tracer,mod_type,''.join(str(x) for x in savenum))
        dd   = az.from_netcdf(os.path.join(mc_dir, n))
        dd_comp[w] = dd
        
        trace_summary = az.summary(dd, round_to=6)
        trace_summary_comp[w] = trace_summary
    except FileNotFoundError:
        n    = '{}.{}.{}.{}.netcdf'.format('PLM1',tracer,mod_type,J)
        dd   = az.from_netcdf(os.path.join(mc_dir, n))
        trace_summary_comp[w] = az.summary(dd, round_to=6)
 
        dd_comp[w] = dd
        trace_summary_comp[w] = trace_summary


def to_hdi(trace_arr):
    cl, ch = az.hdi(trace_arr, 0.95)
    #cl, ch = az.hdi(trace_arr, 0.99)
    mask = np.where((trace_arr >= cl) & (trace_arr <= ch), True, False)
    return trace_arr[mask]



# Variables to plot
var_names_map = {'piston': ['tau1'],
                 'exponential': ['tau1',],
                 'exp_pist_flow': ['tau1','eta1'],
                 'dispersion': ['tau1','D1'],
                 'exp_pist_flow-exp_pist_flow': ['tau1','tau2','f1','f2','eta1','eta2','tau'],
                 'exp_pist_flow-piston': ['tau1','tau2','f1','f2','eta1','tau','J','thalf_cfc'],
                 'dispersion-piston': ['tau1','tau2','f1','f2','D1','tau','J','thalf_cfc']}

var_names = []
for i in trace_summary.index:
    if i in var_names_map[mod_type]:
        var_names.append(i)






#-------------
#
# Posterior Predictive Concentration Plots
# Should match what the actual MCMC used
#
tracers   = tracer.split('.')



# Double loop so this matches the ens_dict structure
C_preds = {}
for t in tracers:
    C_preds[t] = {}
    for w in wells:
        dd = copy.deepcopy(dd_comp)[w]
        trace_summary = copy.deepcopy(trace_summary_comp)[w]
        # Initialize Models
        if t in ['H3','He3']:
            tr_in = copy.deepcopy(C_in_dict)['H3']
        else:
            tr_in = copy.deepcopy(C_in_dict)[t]
        conv_ = conv.tracer_conv_integral(C_t=tr_in, t_samp=tr_in.index[-1]) # first  comp of bmm
    
        # Setup parameters
        pp = {'tau1':False,'tau2':False,'f1':False,'f2':False,
              'eta1':False,'eta2':False,'J':False,
              'D1':False,'D2':False,
              'thalf_cfc':False, 'lamsf6':False} # intialize parameters
        vrs  = list(trace_summary.index)
        # Loop Through Posterior samples
        C_ens = [] # single trace at single well
        for s in range(np.asarray(dd['posterior']['tau1']).shape[1]):
        #for s in range(200):
            for v in vrs:
                pp[v]  = np.asarray(dd['posterior'][v])[0,:][s]

            tau1, tau2, f1, f2, J = pp['tau1'], pp['tau2'], pp['f1'], pp['f2'], pp['J']
            eta1, eta2, D1, D2 =  pp['eta1'], pp['eta2'], pp['D1'], pp['D2']
            thalf_cfc = pp['thalf_cfc']
            
            # First, set up convolution parameters
            if t == 'H3':
                conv_.update_pars(tau=tau1, mod_type=mod_type1, t_half=12.34, eta=eta1, D=D1, bbar=False, Phi_im=False)
            elif t == 'He3':
                conv_.update_pars(tau=tau1, mod_type=mod_type1, t_half=12.34, rad_accum='3He', eta=eta1, D=D1, bbar=False, Phi_im=False)
            elif t == 'He4_ter':
                if 'J' not in vrs:
                    J = np.log10(ng_utils.J_flux(Del=1., rho_r=2700, rho_w=1000, U=3.0, Th=10.0, phi=0.05)) # sample of these parameters?
                conv_.update_pars(tau=tau1, mod_type=mod_type1, rad_accum='4He', J=10**J, eta=eta1, D=D1, bbar=False, Phi_im=False)
            else:
                conv_.update_pars(tau=tau1, mod_type=mod_type1, eta=eta1, D=D1, bbar=False, Phi_im=False)
            if t=='CFC12' and 'thalf_cfc' in vrs:
                conv_.thalf_2_lambda(thalf_cfc)
            # Convolution to find model 1 concentration
            C1_ = conv_.convolve()
            # Update for model 2, skip if using single model
            if mod_type2:
                conv_.mod_type = mod_type2
                conv_.tau      = tau2
                conv_.eta      = eta2
                conv_.D        = D2
                C2_ = conv_.convolve()
            else:
                f1 = 1.0
                C2_ = 0.0 
            # Use fraction factors for weighted sum of concentrations
            C_ = C1_*f1 + C2_*f2
            
            ## Account for CFC degredation
            #if v=='lamcfc' and t=='CFC12':
            #    C_ *= (1-pp['lamcfc'])
            ## Account for SF6 contamination
            if t=='SF6' and 'lamsf6' in vrs:
                C_ *= (1+pp['lamsf6'])
            
            C_ens.append(C_)
        C_preds[t][w] = np.array(C_ens)















# Plot it...
# Plot all the wells together
ll = ['A','B','C']
cm = ['C0','C1','C3']

if len(tracers)==2:
    fig, axes = plt.subplots(len(wells), len(tracers), figsize=(len(tracers)+2,len(wells)+1.2))
    fig.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.05, wspace=0.75)
    plt.rcParams['ytick.major.pad']=4.0

if len(tracers)==3:
    fig, axes = plt.subplots(len(wells), len(tracers), figsize=(len(tracers)+2.5,len(wells)+1.2))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.05, wspace=1.05)
    plt.rcParams['ytick.major.pad']=4.0

if len(tracers)==4:
    fig, axes = plt.subplots(len(wells), len(tracers), figsize=(len(tracers)+3,len(wells)+0.0))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.05, wspace=1.15)
    plt.rcParams['ytick.major.pad']=4.0



for j,w in zip(np.arange(len(wells)), wells):
    for i,t in zip(np.arange(len(tracers)), tracers):
        ax = axes[j,i]
        
        if t == 'He4_ter':
            obs = copy.deepcopy(ens_dict)[t][w].to_numpy().ravel() * 1.e8
            sim = to_hdi(copy.deepcopy(C_preds)[t][w] * 1.e8)
        else:
            obs = copy.deepcopy(ens_dict)[t][w].to_numpy().ravel() 
            sim = to_hdi(copy.deepcopy(C_preds)[t][w])
        # Get rid of contaminated samples
        if 'SF6'==t and w=='PLM6':
            #sns.kdeplot(y=sim, ax=ax, color='C{}'.format(j), fill='C{}'.format(j), linewidth=0.1, alpha=0.01)
            sns.kdeplot(y=sim, ax=ax, color=cm[j], fill=cm[j], linewidth=0.75, alpha=0.01)
            sns.kdeplot(y=obs, ax=ax, color='C2', fill='C2', linewidth=0.75, linestyle='--', alpha=0.01)
            ax.text(0.62, 0.5, 'Contam.', bbox=dict(facecolor='white',alpha=0.65), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            #ax.set_yticks([])
            #ax.set_yticklabels([])
            ax.set_xlim(0.0, ax.get_xlim()[1]*0.05)
        else:
            sns.kdeplot(y=sim, ax=ax, color=cm[j], fill=cm[j], linewidth=1.5)
            sns.kdeplot(y=obs, ax=ax, color='C2', fill='C2', linewidth=1.5, linestyle='--')
            ax.set_xlim(0.0, ax.get_xlim()[1]*0.98)
        # Cleanup
        if j == 1:
            if t == 'He4_ter':
                ax.set_ylabel('{} ({})'.format(ylabs[t], ylabs_units[t]), labelpad=10.0)
            else:
                ax.set_ylabel('{} ({})'.format(ylabs[t], ylabs_units[t]))
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel('')
        ax.yaxis.set_major_locator(MaxNLocator(3)) 
        ax.minorticks_on()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        #ax[i].tick_params(bottom=True, top=False, which='both', labelrotation=45, pad=0.025)
    axes[j,0].text(0.52, 0.90, w.split('_')[0], 
                   horizontalalignment='center', verticalalignment='center', transform=axes[j,0].transAxes)
        
    if not mod_type2:
        mt = mod_type1
    elif mod_type2:
        mt = '{}-{}'.format(mod_type1,mod_type2)
plt.savefig(os.path.join(fdir, 'post_ppred.png'), dpi=300)
plt.savefig(os.path.join(fdir, 'post_ppred.svg'), format='svg')
plt.show()








