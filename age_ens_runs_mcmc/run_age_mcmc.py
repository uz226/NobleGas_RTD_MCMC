#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 12:42:46 2021

@author: nicholasthiros
"""

# Update 12/10/2021
# Can jointly include multiple tracer concentrations in the likelihood function


import numpy as np
import pandas as pd
import pickle
import sys
import copy
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, MaxNLocator, LogLocator)
import matplotlib.ticker as ticker
plt.rcParams['font.size']=14

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


#-----
# MCMC inversion info
try:
    sys.argv[1]
    wells = [str(sys.argv[1])]
except IndexError:
    wells     = ['PLM1','PLM7','PLM6'] 
    #wells     = ['PLM6']
    pass

#-----

#mod_type1 = 'piston'         # must have a mod_type1
mod_type1 = 'exponential'
#mod_type1 = 'dispersion'
#mod_type1  = 'exp_pist_flow'

mod_type2  = False            # mod_type2 = False is a single component RTD
#mod_type2  = 'piston'
#mod_type2  = 'exponential'
#mod_type2  = 'dispersion'
#mod_type2  = 'exp_pist_flow'

#-----

savedir   = 'conv_traces'
try:
    sys.argv[2]
    tracers = [str(sys.argv[2])]
except IndexError:
    #tracers   = ['CFC12']
    #tracers   = ['H3']
    #tracers   = ['SF6']
    tracers   = ['He4_ter']
    #tracers   = ['CFC12','SF6','H3','He4_ter'] 
    pass

#-----

savenum  = [0]    # constant J, no cfc decay, no sf6 contam.
#savenum  = [1]    # variable J
#savenum  = [2]    # cfc decay
#savenum  = [3]    # sf6 contam.

#savenum  = [1,2]    # variable J and cfc decay
#savenum  = [1,2,3]  # variable J and cfc decay and sf6 contam
#savenum  = [2,3]    # constant J and cfc decay and sf6 contam

#-----

# some additional observation error - these are percent errors as decimals 
obs_perr_dict = {}
for w in ['PLM1','PLM7','PLM6']:
    # put a 5 percent error on all the wells
    #obs_perr_dict[w] = {'CFC12':  0.15,
    #                    'SF6':    0.15,
    #                    'H3':     0.10,
    #                    'He3':    0.15,
    #                    'He4_ter':0.10}
    obs_perr_dict[w] = {'CFC12':  0.05,
                        'SF6':    0.05,
                        'H3':     0.05,
                        'He3':    0.05,
                        'He4_ter':0.05}
# PLM6 has highly contaminated SF6
obs_perr_dict['PLM6']['SF6'] = 10.0



#-----



for ww in wells:
    #-----
    # MCMC inversion info
    # 
    # Check if wells exist for the tracers
    for t in tracers:
        if ww not in map_dict[t].index.to_list():
            print ('no {} at {}'.format(t,ww))
    print ('\nWorking on... {} {} {}-{} tag:{}'.format(ww, tracers, mod_type1, mod_type2, savenum))
    
    #----
    # set up observations
    okw  = {}
    for tt in tracers:
        okw[tt] = {}
        okw[tt]['obs_df'] = ens_dict[tt][ww]
        #okw[tt]['obs_perr'] = 0.10 # the observation errors are now included in ens_dict. Set in age_modeling_mcmc.prep.py
        okw[tt]['obs_perr']  = obs_perr_dict[ww][tt]
        
    #----
    # set up priors 
    # single RTD model
    pkw = {}
    pkw['tau1_low']  =  1.0      
    pkw['tau1_high'] =  1000.0   
    par_names = ['tau1']
               
    # add in second RTD for BMM
    if mod_type2:
        pkw['tau2_low']  = 50.0       
        pkw['tau2_high'] = 15000.0     
        par_names += ['tau2']
        
        pkw['f1_low']  = 0.01      
        pkw['f1_high'] = 0.99     
        par_names += ['f1']
        par_names += ['f2']
    
    if mod_type1 == 'exp_pist_flow':
        pkw['eta1_low']  = 1.0
        pkw['eta1_high'] = 5.0    
        par_names += ['eta1']
    
    if mod_type2 == 'exp_pist_flow':
        pkw['eta2_low']  = 1.0
        pkw['eta2_high'] = 5.0  
        par_names += ['eta2']
        
    if mod_type1 == 'dispersion':
        pkw['D1_low']  = 0.01
        pkw['D1_high'] = 2.00    
        par_names += ['D1']
        
    if mod_type2 == 'dispersion':
        pkw['D2_low']  = 0.01
        pkw['D2_high'] = 2.0    
        par_names += ['D2']
        
    # add variable helium production
    if 'He4_ter' in tracers and 1 in savenum:
        pkw['J_mu'] = np.log10(ng_utils.J_flux(Del=1., rho_r=2700, rho_w=1000, U=3.7, Th=10.2, phi=0.05))
        pkw['J_sd'] = 0.33
        #pkw['J_sd'] = 0.001
        par_names += ['J'] 
        
    # add cfc degredation
    if 'CFC12' in tracers and 2 in savenum:
        pkw['cfc_thalf_lo'] = 5.0 # half-life in years, see Hinsby 2007 WRR
        pkw['cfc_thalf_hi'] = 35.0
        par_names += ['thalf_cfc']
        
    # add sf6 contamination
    if 'SF6' in tracers and 3 in savenum:
        #pkw['lamsf6'] = True
        par_names += ['lamsf6']
    
    pkw['par_names'] = par_names
    
    #----
    # set up convolution hyperparameters
    ckw = {}
    ckw['mod_type1']  = mod_type1
    ckw['mod_type2']  = mod_type2
    for tt in tracers:
        ckw[tt] = {}
        
        if tt in ['CFC11','CFC12','CFC113','SF6']:
            ckw[tt]['C_t']       = copy.deepcopy(C_in_dict[tt])           
            
        elif tt in ['He4_ter']:
            ckw[tt]['C_t']       = copy.deepcopy(C_in_dict[tt])*0.0
            ckw[tt]['rad_accum'] = '4He'
            # 2/7/22 Considered uncertain MCMC parameters
            #ckw[tt]['J']         = ng_utils.J_flux(Del=1., rho_r=2700, rho_w=1000, U=3.0, Th=10.0, phi=0.05)
                
        elif tt in ['He3']:
            ckw[tt]['C_t']       = copy.deepcopy(C_in_dict['H3'])
            ckw[tt]['t_half']    = 12.34
            ckw[tt]['rad_accum'] = '3He'

        elif tt in ['H3']: 
            ckw[tt]['C_t']       = copy.deepcopy(C_in_dict['H3'])
            ckw[tt]['t_half']    = 12.34

    
    #-----
    # Run it
    mc_conv = age_mcmc.conv_mcmc(ww, tracers, okw, ckw, pkw, savedir, ''.join(str(x) for x in savenum))
    #mc_prior = mc_conv.sample_prior()
    mc_out = mc_conv.sample_mcmc()
    #mc_conv.plots()

    trace_summary = az.summary(mc_out, round_to=6)
    print (trace_summary)
    
    
    
    
    
    

make_some_plots = True
if make_some_plots:
    
    mc_conv.plots()
    
    #-----------------------
    # Posterior Predictive
    #-----------------------
    dd = mc_out.copy()
    
    # Double loop so this matches the ens_dict structure
    C_preds = {}
    for t in tracers:
        C_preds[t] = {}
        for w in wells:
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
                        J = np.log10(ng_utils.J_flux(Del=1., rho_r=2700, rho_w=1000, U=3.7, Th=10.2, phi=0.05)) # sample of these parameters?
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
               'nu':r'$\it{\nu}$ (-)'}
    
    
    for w in wells:
        fig, axes = plt.subplots(1,len(tracers),figsize=(len(tracers)+2,3))
        fig.subplots_adjust(left=0.15, right=0.98, top=0.90, bottom=0.05, wspace=0.75)
        for i,t in zip(np.arange(len(tracers)), tracers):
            
            if len(tracers) == 1:
                ax = axes
            else:
                ax = axes[i]
            
            if t == 'He4_ter':
                obs = copy.deepcopy(ens_dict)[t][w].to_numpy().ravel() * 1.e8
                sim = copy.deepcopy(C_preds)[t][w] * 1.e8
            else:
                obs = copy.deepcopy(ens_dict)[t][w].to_numpy().ravel() 
                sim = copy.deepcopy(C_preds)[t][w] 
            sns.kdeplot(y=sim, ax=ax, color='C{}'.format(i), fill='C{}'.format(i), linewidth=1.5)
            sns.kdeplot(y=obs, ax=ax, color='grey', fill='grey', linewidth=1.5, linestyle='--')
            # Cleanup
            ax.set_ylabel('{} ({})'.format(ylabs[t], ylabs_units[t]), labelpad=2)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel('')
            ax.yaxis.set_major_locator(MaxNLocator(4)) 
            ax.minorticks_on()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            #ax[i].tick_params(bottom=True, top=False, which='both', labelrotation=45, pad=0.025)
            fig.suptitle(w.split('_')[0])
        if not mod_type2:
            mt = mod_type1
        elif mod_type2:
            mt = '{}-{}'.format(mod_type1,mod_type2)
        fig.tight_layout()
        #plt.savefig(os.path.join(fdir, 'post_ppred.png'), dpi=300)
        #plt.savefig(os.path.join(fdir, 'post_ppred.svg'), format='svg')
        plt.show()


