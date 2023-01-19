#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:13:30 2021

@author: nthiros
"""


# Script to perform MCMC analysis on dissolved noble gases.
# Estimates recharge temperatures and elevations and excess air parameters
# Generates plots of posterior parameter distributions


import numpy as np
import pandas as pd
import pickle
import scipy
import pdb
import sys

sys.path.insert(0, '../utils')
import noble_gas_utils as ng

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import (MaxNLocator, MultipleLocator, LogLocator)
plt.rcParams['font.size'] = 14
import matplotlib.style as style
style.use('tableau-colorblind10')

import pymc3 as mc
import seaborn as sb
import theano
import theano.tensor as tt
#from theano import as_op
from theano.compile.ops import as_op
import os
import arviz as az

import seaborn as sns


  


#-----------------------------------------
# Temperature Elevation Lapse Rate
#-----------------------------------------
def find_T_lapse(E_lapse, E_lapse_int, lapse_slope):
    """Solve for temperature as function of elevation.
       Assumes lapse rate with temp on x axis versus elevation on y axis.
       Find lapse rate slope and intercept using the weather station data in SCGSR/modleing/MET dir.
    
       Returns: 
            - Temperature (C)
       Inputs: 
           - E_lapse is elevation (m) to find temperature at (or list of elevations)
           - E_lapse_int is the y-intercept (elevation in m) of the lapse rate, ie. elevation where T=0.0
           - lapse_slope is the slope of lapse rate in meters/celcius, ie. -100 --> 100m per 1C)
       """
    T_mle = (E_lapse-E_lapse_int)/lapse_slope
    return T_mle





#--------------------------------
# Read in the Observation Data
#--------------------------------
dd_ = pd.read_excel('../Field_Data/PLM_NobleGas_2021.xlsx', skiprows=1, index_col='SiteID', nrows=5)
dd  = dd_.copy()[['4He','Ne','Ar','Kr','Xe']]
dd.rename(columns={'4He':'He'}, inplace=True)
dd.dropna(inplace=True)
dd.index.name = ''

obs_dict_all = dd.T.to_dict()

well_elev = {'PLM1':2786.889893,
             'PLM6':2759.569824,
             'PLM7':2782.550049}





#-------------------------------
#
# MCMC input args
#
#-------------------------------

# pick a well
well = 'PLM1'    
#well = 'PLM7'
#well = 'PLM6'


Run_Calib = True   # False to only make plots
bckend    = str(well)
 

obs_dict = obs_dict_all[well]
# percent errors -- assume as lab analytical errors
#err_dict={'He':1.5, 'Ne':1.5, 'Ar':2.5, 'Kr':3.1, 'Xe':3.1}  # from Visser et al 2014 "Intercomparison of tritium and noble gases..."
err_dict={'He':1.5, 'Ne':1.5, 'Ar':2.5, 'Kr':3.1, 'Xe':15.1}  # avoid Xe depletion by weighting less

gases = ['Ne', 'Ar', 'Kr', 'Xe']  # gases to use in the mcmc parameter inference


par_nam = ['Ae', 'F', 'E', 'T', 'b', 'm']  # parameter keys


# --- Change the priors below ---

# lapse rate slope (m/C)  
lapse_slope     = -146.0 
err_lapse_slope =  17.0
mmin = lapse_slope - err_lapse_slope
mmax = lapse_slope + err_lapse_slope

print ('lapse rate slope = {:.3f} (C/km)'.format(1/lapse_slope * 1000))
print ('intercept Delta for 2C change = {:.3f} (meters)'.format(lapse_slope*2))
temp_offset = lapse_slope*0 # depress the atmpospheric lapse rate by 1 C -- update: this is handled in the prior distrubtion below now

# lapse rate intercept (m)
lapse_b     = 3354.0 + temp_offset
err_lapse_b = abs(lapse_slope)*2.5  # assume that recharge temperautes within +/- 2.5 degrees C of mean atmospheric temp
bmin = lapse_b - err_lapse_b
bmax = lapse_b + err_lapse_b

# initial excess air volume  (cm3STP/g)
Amin = 1.0e-4
Amax = 1.0e-1

# excess air fractionation parameter (-)
Fmin = 1.0e-1
Fmax = 1.0e+1

# recharge elevations (m above sea level)
Emin = well_elev[well]-10.0 # well screen
Emax = 3300.0 # elevation of top of Snodgrass minus the 100 m water level depth

# recharge temperatures (C)
Tmin = find_T_lapse(Emax, bmin, mmax)
Tmax = find_T_lapse(Emin, bmax, mmax)





#---
# Create a new directory for figures
fdir = '{}_figs'.format(well)
if os.path.exists(fdir) and os.path.isdir(fdir):
    pass
else:
    os.makedirs(fdir)


#---
# variable mapping to names for plots
var_map = {'m':'slope (m/C)',
           'b':'int (m)',
           'Ae':r'A$_{e}$ (cm$^{3}$STP/g)',
           'F':'F (-)',
           'E':'E (m)',
           'T':'T (C)'}

var_map = {'m':'$\it{m}$ (m/C)',
           'b':'$\it{b}$ (m)',
           'Ae':r'$\it{A_{e}}$ (cm$^{3}$STP/g)',
           'F':'$\it{F}$ (-)',
           'E':'$\it{Z}$ (m)',
           'T':'$\it{T_{r}}$ (C)'}




#---------------------- 
# MCMC Beta Priors
#----------------------
par_min  = np.array([np.log10(Amin), np.log10(Fmin), Emin, Tmin, bmin, mmin]) #logAe, logF, E, T, int, slope
par_max  = np.array([np.log10(Amax), np.log10(Fmax), Emax, Tmax, bmax, mmax]) #logAe, logF, E, T, int, slope
par_bnd  = pd.DataFrame(data=np.column_stack((par_min,par_max)), index=par_nam, columns=['min','max'])
par_bnd_ = par_bnd.T.to_dict()

def frombeta(plist, blist):
    '''plist is a list of strings with variables that can include ['Ae','F','E','T']  
       blist is a list of beta distribution values that correspond to blist
       Returns list in same order of plist and blist that has parameter values converted from beta dist. to normal space'''
    vals_norm = blist.copy()
    for i in range(len(plist)):
        p = plist[i]
        v = blist[i]
        # convert from beta distribution to normal space
        vals_norm[i] = v * (par_bnd_[p]['max']-par_bnd_[p]['min'])+par_bnd_[p]['min']
    return vals_norm

        
#--------------------- 
# MCMC Model
#---------------------
@as_op(itypes=[tt.dvector], otypes=[tt.dvector])
def ce_exc_wrapper(theta):
    #pdb.set_trace()
    Ae_log, F_log, E, T = theta
    Ae, F = 10**Ae_log, 10**F_log

    ce_mod  = ng.noble_gas_fun(gases=gases, E=E, T=T, Ae=Ae, F=F, P='lapse_rate').ce_exc(add_eq_conc=True)
    ce_mod_ = np.array([ce_mod[i] for i in gases]) 
    return ce_mod_
    

def mcmc_model():
    model = mc.Model()
    with model:
        #pdb.set_trace()
        #--- 
        # Priors
        #
        # Beta priors for Ae, F, and E
        Ae_beta = mc.Beta('Ae_beta', alpha=2, beta=2) 
        F_beta  = mc.Beta('F_beta',  alpha=2, beta=2)
        E_beta  = mc.Beta('E_beta',  alpha=2, beta=4)
        
        # Convert from Beta [0,1] to parameter bounds
        Ae = mc.Deterministic('Ae', Ae_beta * (par_bnd_['Ae']['max']-par_bnd_['Ae']['min'])+par_bnd_['Ae']['min'])
        F  = mc.Deterministic('F',  F_beta  * (par_bnd_['F']['max']-par_bnd_['F']['min'])+par_bnd_['F']['min'])
        E  = mc.Deterministic('E',  E_beta  * (par_bnd_['E']['max']-par_bnd_['E']['min'])+par_bnd_['E']['min'])
        
        # Normal priors for lapse rate
        m = mc.Normal('m', mu=lapse_slope, sigma=err_lapse_slope)
        #b = mc.Normal('b', mu=lapse_b,     sigma=err_lapse_b)
        
        b_beta = mc.Beta('b_beta', alpha=2, beta=2.5)
        b  = mc.Deterministic('b',  b_beta  * (par_bnd_['b']['max']-par_bnd_['b']['min'])+par_bnd_['b']['min'])
                
        # Find T using lapse rate and elevation 
        T_ = (E-b)/m
        #T_ = tt.switch(T_<0.0, 0.01, T_)  # set a negative recharge temperature to 0
        T  = mc.Deterministic('T', T_)
        
        # Parameter vector to pass to CE model
        theta = tt.as_tensor_variable([Ae, F, E, T])
            
        # Prior for student-t degrees of freedom (nu)
        nu_ = mc.Beta('nu_', alpha=2.0, beta=0.1) # in interval [0,1]
        nu  = mc.Deterministic('nu', nu_*(30.0-1.0)+1.0) # degrees of freedom now between [1,30]
        
        #---
        # Observations
        obs_mu = np.array([obs_dict[i] for i in gases])
        # These are percent error
        obs_sd  = obs_mu * np.array([err_dict[i]/100 for i in gases])
    
        #---
        # Forward model CE model output Concentration
        mod_out = ce_exc_wrapper(theta)
         
        #---
        # Likelihood -- assuming normal observation error model
        #like = mc.Normal('like', mu=mod_out, sd=obs_sd, observed=obs_mu)
        # Update to student-t distribution
        like = mc.StudentT('like', mu=mod_out, sigma=obs_sd, nu=nu, observed=obs_mu) 
    return model



#---------------------------
# Prior Predictive
#---------------------------
with mcmc_model():
    # This does not include Helium
    prior_checks = mc.sample_prior_predictive(samples=10000, random_seed=10)
    
    
    # --- Plotting ---
    
    # Convert Ae and F from log-beta space
    for p in ['Ae','F']:
        prior_checks[p] = 10**prior_checks[p]
          
    # Parameter distributions
    var_names = ['m','b','Ae','F','E','T'] 
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(6,4))
    fig.subplots_adjust(left=0.03, right=0.98, top=0.98, bottom=0.2, hspace=0.85, wspace=0.15)
    for i in range(len(var_names)):
        r,c = i//3, i%3
        dd = prior_checks[var_names[i]]
        if var_names[i] in ['Ae','F']:
            #ax[r,c].set_xscale('log')
            bins=np.logspace(np.log10(dd.min()),np.log10(dd.max()), 30)
            ax[r,c].set_xscale('log')
            ax[r,c].xaxis.set_major_locator(LogLocator(base=10, numticks=10))
            #ax[r,c].axvline(np.median(dd), color='black', linestyle='--', zorder=10)
        else:
            bins = 30
            ax[r,c].xaxis.set_major_locator(MaxNLocator(nbins=4))
            #ax[r,c].axvline(np.median(dd), color='black', linestyle='--', zorder=10)
        ax[r,c].minorticks_on()
        ax[r,c].hist(dd, bins=bins, density=False, alpha=0.4)
        ax[r,c].set_xlabel(var_map[var_names[i]], labelpad=0.5)
        #
        if var_names[i] == 'b':
            ax[r,c].axvline(lapse_b, color='C0', linestyle='-', linewidth=2.0)
            ax[r,c].axvline(lapse_b+lapse_slope,   color='C1', linestyle='--', linewidth=2.0)
            ax[r,c].axvline(lapse_b+lapse_slope*2, color='C1', linestyle='--', linewidth=2.0)
            ax[r,c].axvline(lapse_b-lapse_slope,   color='C2', linestyle='-.', linewidth=2.0)
            ax[r,c].axvline(lapse_b-lapse_slope*2, color='C2', linestyle='-.', linewidth=2.0)
        if var_names[i] == 'T':
            ax[r,c].axvline(find_T_lapse(2920, lapse_b, lapse_slope), color='C0', linestyle='-', linewidth=2.0)
            ax[r,c].axvline(find_T_lapse(2920, lapse_b+lapse_slope,   lapse_slope),   color='C1', linestyle='--', linewidth=2.0)
            ax[r,c].axvline(find_T_lapse(2920, lapse_b+lapse_slope*2, lapse_slope),   color='C1', linestyle='--', linewidth=2.0)
            ax[r,c].axvline(find_T_lapse(2920, lapse_b-lapse_slope,   lapse_slope),   color='C2', linestyle='-.', linewidth=2.0)
            ax[r,c].axvline(find_T_lapse(2920, lapse_b-lapse_slope*2, lapse_slope),   color='C2', linestyle='-.', linewidth=2.0)
        if var_names[i] == 'E':
            pass
            #ax[r,c].axvline(2920, color='C0', linestyle='-', linewidth=2.0)
        # Cleanup
        ax[r,c].spines[['right', 'left', 'top']].set_visible(False)
        ax[r,c].tick_params(axis='y',which='both',left=False,labelleft=False)
        ax[r,c].tick_params(axis='x',which='both',bottom=True, labelrotation=45, pad=0.1)
    #fig.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), fdir, 'ng_prior_parameters.png'), dpi=300)
    plt.savefig(os.path.join(os.getcwd(), fdir, 'ng_prior_parameters.svg'), format='svg')
    plt.show()



    # Joint Prior Distributions
    plt.rcParams['figure.subplot.hspace']=0.05
    plt.rcParams['figure.subplot.wspace']=0.05
    plt.rcParams['figure.subplot.top']=0.99
    plt.rcParams['figure.subplot.bottom']=0.15
    plt.rcParams['figure.subplot.right']=0.99
    plt.rcParams['figure.subplot.left']=0.19
    plt.rcParams['axes.labelpad']=24
    fs = 16 # fontsize
    
    kde_kwargs = {"contourf_kwargs": {"alpha": 1,"cmap":'Blues'}, "contour_kwargs": {'alpha':0.1,}}
    #ax = az.plot_pair(idata, kind='kde', marginals=True, var_names=var_names, 
    #                  kde_kwargs=kde_kwargs, figsize=(14,10))
    ax = az.plot_pair(prior_checks, kind='hexbin', marginals=False, var_names=var_names, figsize=(9,7)) #figsize=(14,10))
    #kde_kwargs = {"contourf_kwargs": {"alpha": 0.0}, "contour_kwargs": {'alpha':0.0,}}
    #marg_kwargs = {"color":'C1'}
    #mg = az.plot_pair(idata, group='prior', kind='kde', fill_last=True, marginals=True, var_names=var_names, 
    #                 kde_kwargs=kde_kwargs, marginal_kwargs=marg_kwargs, ax=ax)
    for ii in range(len(ax)):
        #ax[ii][0].set_ylabel(lmp[ ax[ii][0].get_ylabel()], fontsize=fs, rotation=45)
        ax[ii][0].set_ylabel(var_map[ax[ii][0].get_ylabel()], fontsize=fs, rotation=45)
        ax[ii][0].yaxis.labelpad = 30
    for jj in range(len(ax[-1])):
        #ax[-1][jj].set_xlabel(lmp[ax[-1][jj].get_xlabel()], fontsize=fs, rotation=45)
        ax[-1][jj].xaxis.labelpad = 5
        ax[-1][jj].set_xlabel(var_map[ax[-1][jj].get_xlabel()], fontsize=fs, rotation=0)
        ax[-1][jj].set_xticklabels(ax[-1][jj].get_xticks().round(2), rotation = 45)
    #plt.savefig(os.path.join(os.getcwd(), fdir, 'posterior_joint.png'), dpi=300)
    #plt.savefig(os.path.join(os.getcwd(), fdir, 'posterior_joint.svg'), format='svg')
    plt.show()

    
    # Concentration Predictions
    pr = prior_checks['like']
    # plot to check
    fig, ax = plt.subplots(ncols=4, figsize=(6,4))
    for i in range(len(gases)):
        # Observations
        ax[i].scatter(0.0, np.array([obs_dict[j] for j in gases])[i], color='red', zorder=10)
        #ax[i].errorbar(0.25, obs_arr[i], yerr=obs_err[i], color='red', zorder=10)
        # Predictions using boxplots
        ax[i].boxplot(pr[:,i], positions=[0], showmeans=False, medianprops={'color':'black','linewidth':2})
        # Clean up
        ax[i].set_xticks([])
        ax[i].set_xticklabels([])
        ax[i].set_xlabel(gases[i])
        #ax[i].yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax[i].set_yscale('log')
        ymin = np.floor(np.log10(ax[i].get_ylim()[0]))
        ymax = np.ceil(np.log10(ax[i].get_ylim()[1]))
        ax[i].set_ylim(10**ymin, 10**ymax)
        #
        ax[i].xaxis.labelpad = 5
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        #ax[i].spines['bottom'].set_visible(False)
        ax[0].set_ylabel(r'Concentration (cm$^{3}$STP/g)')
    #fig.suptitle('Prior Predictive')
    fig.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), fdir, 'prior_predictive.png'), dpi=300)
    plt.show()


    



#------------------------------------------------
# Sample the posterior
#------------------------------------------------
if Run_Calib:
    from shutil import rmtree
    if os.path.exists(os.path.join('./traces/',bckend+'.netcdf')):
        #rmtree(bckend+'.netcdf')
        os.remove(os.path.join('./traces/',bckend+'.netcdf'))
        
    with mcmc_model():
        #prop_var = np.eye(par.shape[1]) * np.ones(par.shape[1])*0.25
        sampler = mc.DEMetropolisZ(tune_interval=5000)#, S=prop_var) 
        #sampler = mc.Metropolis()
        #sampler = mc.Slice()
        
        trace = mc.sample(tune=10000, draws=50000, step=sampler, chains=4, cores=1, random_seed=123423, 
                          idata_kwargs={'log_likelihood':False}, discard_tuned_samples=True)
        
        #post_pred = mc.sample_posterior_predictive(trace, samples=1000, random_seed=123423)
        #idata = az.from_pymc3(trace, prior=prior_checks, posterior_predictive=post_pred, log_likelihood=False)
        idata = az.from_pymc3(trace, prior=prior_checks, log_likelihood=False)
    
        # Save it
        az.to_netcdf(idata, os.path.join('./traces/',bckend+'.netcdf'))
else:
    pass



#------------------------------------------------
# Plotting MCMC Results
#------------------------------------------------
with mcmc_model():
    # Read in data
    idata = az.from_netcdf(os.path.join('./traces/',bckend+'.netcdf')) 
 
    # Convert from beta parameters to normal space
    #for p in ['E']:
    #    idata['posterior'][p] = idata['posterior'][p] * (par_bnd_[p]['max']-par_bnd_[p]['min'])+par_bnd_[p]['min']
    for p in ['Ae','F']:
        idata['posterior'][p] = 10**(idata['posterior'][p])
    
    # Expand the nus
    #for i in range(len(gases)):
    #    idata['posterior']['nu{}'.format(gases[i])] = idata.posterior.nu[:,:,i]
    
 
    # save this again for later use...
    az.to_netcdf(idata, os.path.join('./traces/',bckend+'_trans.netcdf')) 
 
    # Trace Summary
    trace_summary = az.summary(idata, round_to=6)
    print (trace_summary)
    
    #
    # Traceplot
    var_names = ['m','b','Ae','F','E','T','nu']
    ax = az.plot_trace(idata, var_names=var_names)
    for ii in range(len(ax)):
        for jj in range(len(ax[ii])):
            t = ax[ii][jj].get_title()
            try:    
                #ax[ii][jj].tick_params(axis='x', pad=0.1)
                #ax[ii][jj].set_xlabel(var_map[t], fontsize=12)
                ax[ii][jj].set_title(var_map[t], fontsize=14)
            except KeyError:
                pass
            #ax[ii][jj].set_title('')
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), fdir, 'trace_plot.png'), dpi=300)
    plt.show()
    
    
    ## Plot nu seperate
    ## Expand the nus
    #for i in range(len(gases)):
    #    idata['posterior']['nu{}'.format(gases[i])] = idata.posterior.nu[:,:,i]
    # 
    # ax = az.plot_trace(idata, var_names=var_names)
    #for ii in range(len(ax)):
    #    for jj in range(len(ax[ii])):
    #        t = ax[ii][jj].get_title()
    #        try:    
    #            #ax[ii][jj].tick_params(axis='x', pad=0.1)
    #            #ax[ii][jj].set_xlabel(var_map[t], fontsize=12)
    #            ax[ii][jj].set_title(var_map[t], fontsize=14)
    #        except KeyError:
    #            pass
    #        #ax[ii][jj].set_title('')
    #plt.tight_layout()
    ##plt.savefig(os.path.join(os.getcwd(), fdir, 'trace_plot.png'), dpi=300)
    #plt.show()
    
    
    #
    # Posterior plot
    var_names = ['m','b','Ae','F','E','T'] 
    ax = az.plot_posterior(idata, var_names=var_names, round_to=3)
    for ii in range(len(ax)):
        for jj in range(len(ax[ii])):
            t = ax[ii][jj].get_title()
            try:    
                ax[ii][jj].xaxis.labelpad = 5
                ax[ii][jj].set_xlabel(var_map[t], fontsize=16)
                #ax[ii][jj].set_title(var_map[t], fontsize=14)
            except KeyError:
               pass
            ax[ii][jj].set_title('')
    plt.savefig(os.path.join(os.getcwd(), fdir, 'marginal_posterior.png'), dpi=300)
    plt.show()


     
    ##
    ## Posterior plot -- nu
    #var_names = ['nuNe','nuAr','nuKr','nuXe']
    #ax = az.plot_posterior(idata, var_names=var_names, round_to=3)
    #for ii in range(len(ax)):
    #    for jj in range(len(ax[ii])):
    #        t = ax[ii][jj].get_title()
    #        try:    
    #            ax[ii][jj].xaxis.labelpad = 5
    #            ax[ii][jj].set_xlabel(var_map[t], fontsize=16)
    #            #ax[ii][jj].set_title(var_map[t], fontsize=14)
    #        except KeyError:
    #           pass
    #        ax[ii][jj].set_title('')
    ##plt.savefig(os.path.join(os.getcwd(), fdir, 'marginal_posterior.png'), dpi=300)
    #plt.show()
    




    #
    # Marginal Posteriors with priors
    var_names = ['m','b','Ae','F','E','T']
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6,4))
    fig.subplots_adjust(top=0.98,bottom=0.2,left=0.05,right=0.96,hspace=0.85)
    for i in range(len(var_names)):
        r,c = i//3,i%3
        ax = axes[r,c]
        
        v = var_names[i]
        post  = np.asarray(idata.posterior[v]).ravel()
        prior = np.asarray(idata.prior[v]).ravel()
        
        logscl = False
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if v in ['Ae']:
            logscl = True
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
            
        sns.kdeplot(x=post, ax=ax,  bw_adjust=0.25, color='C0', fill='C0', linewidth=1.5, linestyle='-', log_scale=logscl)
        sns.kdeplot(x=prior, ax=ax, bw_adjust=0.25, color='red', linewidth=1.5, linestyle='--', log_scale=logscl)
        
        # Cleaup
        ax.set_xlabel(var_map[var_names[i]], labelpad=0.5)
        ax.minorticks_on()
        ax.tick_params(axis='x',which='both',bottom=True,pad=0.1, labelrotation=45)
        
        ax.tick_params(axis='y',which='both',left=False,labelleft=False)
        ax.set_ylabel('')
        ax.grid(axis='x')
        # Trim out really wide priors -- needs to be done manually
        if v == 'F':
            ax.set_xlim(0.0,1.0)
            if well in ['PLM8','Shumway']:
                ax.set_xlim(0.0,20.0)
        if v == 'Ae':
            ax.set_xlim(0.001,0.1)
            #ax.set_xlim(0.002,0.1)
        #if v == 'T':
        #    ax.set_xlim(0.01, 10)
    plt.savefig(os.path.join(os.getcwd(), fdir, 'posterior_prior.png'), dpi=300)
    plt.show()



    #
    # Joint Posteriors
    plt.rcParams['figure.subplot.hspace']=0.05
    plt.rcParams['figure.subplot.wspace']=0.05
    plt.rcParams['figure.subplot.top']=0.99
    plt.rcParams['figure.subplot.bottom']=0.15
    plt.rcParams['figure.subplot.right']=0.98
    plt.rcParams['figure.subplot.left']=0.19
    plt.rcParams['axes.labelpad']=24
    fs = 16 # fontsize
    
    kde_kwargs = {"contourf_kwargs": {"alpha": 1,"cmap":'Blues'}, "contour_kwargs": {'alpha':0.1,}}
    #ax = az.plot_pair(idata, kind='kde', marginals=True, var_names=var_names, 
    #                  kde_kwargs=kde_kwargs, figsize=(14,10))
    ax = az.plot_pair(idata, kind='hexbin', marginals=False, var_names=var_names, figsize=(9,7)) #figsize=(14,10))
    #kde_kwargs = {"contourf_kwargs": {"alpha": 0.0}, "contour_kwargs": {'alpha':0.0,}}
    #marg_kwargs = {"color":'C1'}
    #mg = az.plot_pair(idata, group='prior', kind='kde', fill_last=True, marginals=True, var_names=var_names, 
    #                 kde_kwargs=kde_kwargs, marginal_kwargs=marg_kwargs, ax=ax)
    for ii in range(len(ax)):
        #ax[ii][0].set_ylabel(lmp[ ax[ii][0].get_ylabel()], fontsize=fs, rotation=45)
        ax[ii][0].set_ylabel(var_map[ax[ii][0].get_ylabel()], fontsize=fs, rotation=45)
        ax[ii][0].yaxis.labelpad = 30
    for jj in range(len(ax[-1])):
        #ax[-1][jj].set_xlabel(lmp[ax[-1][jj].get_xlabel()], fontsize=fs, rotation=45)
        ax[-1][jj].xaxis.labelpad = 5
        ax[-1][jj].set_xlabel(var_map[ax[-1][jj].get_xlabel()], fontsize=fs, rotation=0)
        ax[-1][jj].set_xticklabels(ax[-1][jj].get_xticks().round(2), rotation = 45)
    plt.savefig(os.path.join(os.getcwd(), fdir, 'posterior_joint.png'), dpi=300)
    plt.savefig(os.path.join(os.getcwd(), fdir, 'posterior_joint.svg'), format='svg')
    plt.show()








#-------------------------
# Posterior predictive 
#-------------------------

gases_ = ['Ne', 'Ar', 'Kr', 'Xe', 'He']

#----
# Observations
obs_arr =  np.array([obs_dict[i] for i in gases_])
obs_err = obs_arr * np.array([err_dict[i]/100 for i in gases_])

#----
# Posterior parameter arrays 
m, b, Ae, F, E, T = [trace_summary.loc[p, 'mean'] for p in ['m','b','Ae','F','E','T']]
# The posterior chains -- Ndraws x Mpars
p_ens = np.array([np.asarray(idata.posterior[p]).ravel() for p in ['m','b','Ae','F','E','T']]).T
# Drop parameters below 3% hdi and above 97% hdi
#f1 = np.array([p_ens[:,i] >= trace_summary['hdi_3%'][i] for i in range(len(['m','b','Ae','F','E','T']))]).T
#f2 = np.array([p_ens[:,i] <= trace_summary['hdi_97%'][i] for i in range(len(['m','b','Ae','F','E','T']))]).T
#f = np.concatenate((f1,f2),axis=1).all(axis=1)
#p_ens = p_ens[f]
# Random shuffle 
np.random.shuffle(p_ens) 
# Parameter ensemble arrays for CE mod
MM, BB, AA, FF, EE, TT = [p_ens[:,i] for i in range(len(['m','b','Ae','F','E','T']))]
#----
# Forward predictions of concentrations
# Maximum a posteriori concentrations
C_max_ = ng.noble_gas_fun(gases=gases_, E=E, T=T, Ae=Ae, F=F, P='lapse_rate').ce_exc(add_eq_conc=True)
C_max = np.array([C_max_[i] for i in gases_])
# Ensemble of predictions for uncertainty
C_unc = []
for i in range(20000):
    c = ng.noble_gas_fun(gases=gases_, E=EE[i], T=TT[i], Ae=AA[i], F=FF[i], P='lapse_rate').ce_exc(add_eq_conc=True)
    cc = np.array([c[i] for i in gases_])
    C_unc.append(cc)
C_unc = np.array(C_unc)



#-----------
# Post-predictive concentration box-plots
scale_map = {'He':1.e8,
             'Ne':1.e7,
             'Ar':1.e4,
             'Kr':1.e8,
             'Xe':1.e8}



#-----------
# Post-predictive concentration distributions
# Adds in terrigenic 4He
fig, ax = plt.subplots(ncols=5, figsize=(7,3))
fig.subplots_adjust(left=0.15, right=0.92, top=0.90, wspace=0.8)
for i in range(len(gases_)):
    gg  = gases_[i]
    scl = scale_map[gg]
    # Predictions using boxplots
    sim_ = C_unc[:,i]*scl
    kde = sns.kdeplot(y=sim_, ax=ax[i], shade=True, color='C0', fill='C0', linewidth=1.5, bw_adjust=0.25, label=r'C$_{\mathrm{sim}}$')
    # Observations
    obs_ = np.random.normal(obs_arr[i]*scl, obs_err[i]*scl/1.96, len(C_unc[:,i]))
    sns.kdeplot(y=obs_, ax=ax[i], color='grey', linewidth=1.5, linestyle='--', bw_adjust=0.25, label=r'C$_{\mathrm{obs}}$')
    # Clean up
    ax[i].set_xticks([])
    ax[i].set_xticklabels([])
    ax[i].set_xlabel(r'{} [10$^{{{}}}$]'.format(gases_[i], int(-np.log10(scl))))
    ax[i].xaxis.labelpad = 5
    ax[i].yaxis.labelpad = 5
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    # Terrigenic He
    if gg == 'He':
        sns.kdeplot(y=(obs_-sim_), ax=ax[i], color='C1', fill='C1', linewidth=1.5, linestyle=':', bw_adjust=0.25, label=r'C$_{\mathrm{ter}}$')
        if obs_.max() >= 10.0:
            ax[i].yaxis.set_major_locator(MultipleLocator(2))
            ax[i].yaxis.set_minor_locator(MultipleLocator(1))
        if obs_.max() >= 20.0:
            ax[i].yaxis.set_major_locator(MultipleLocator(5))
            ax[i].yaxis.set_minor_locator(MultipleLocator(1))
        if obs_.max() >= 50.0:
            print (obs_.max())
            ax[i].yaxis.set_major_locator(MultipleLocator(10))
            ax[i].yaxis.set_minor_locator(MultipleLocator(2))
        if obs_.max() < 10.0:
            ax[i].yaxis.set_major_locator(MultipleLocator(1))
            ax[i].yaxis.set_minor_locator(MultipleLocator(0.25))
        # For manuscript...
        ax[i].set_ylim(0,16)
        ax[i].yaxis.set_major_locator(MultipleLocator(2))
        ax[i].yaxis.set_minor_locator(MultipleLocator(1))
        ax[i].set_xlim(0, ax[i].get_xlim()[1]*0.5)
ax[0].set_ylabel(r'Concentration (cm$^{3}$STP/g)')
#fig.suptitle('{} Posterior Predictive'.format(well))
#ax[0].legend()
#fig.tight_layout()
plt.savefig(os.path.join(os.getcwd(), fdir, 'post_conc_pred_kde.png'), dpi=300)
plt.savefig(os.path.join(os.getcwd(), fdir, 'post_conc_pred_kde.svg'), format='svg')
plt.show()







#-----------
# Posterior Predictive lapse rate
# KDE for posterior
from matplotlib.colors import LinearSegmentedColormap
cmap0 = LinearSegmentedColormap.from_list('', ['lightblue','darkblue'])


fig, ax = plt.subplots(figsize=(7,5))
xx = np.linspace(-5, 12, 500)
# prior model
ax.plot(xx, lapse_slope*xx + lapse_b, c='C2', linewidth=3, zorder=5, label='Prior Model')
# maximum a posteriori
ax.plot(xx, trace_summary.loc['m','mean']*xx + trace_summary.loc['b','mean'], 
        linewidth=3, color='black', zorder=6, label='Max a Posteriori')
# error ensembles
for i in range(500):
    ax.plot(xx, MM[i]*xx + BB[i], c='grey', alpha=0.5, label='Posterior Uncertainty' if i==0 else '')
# Add inferred recharge zone
#ax.scatter(trace_summary.loc['T','mean'], trace_summary.loc['E','mean'], zorder=8)
#xmin = trace_summary.loc['T','mean']-trace_summary.loc['T','sd']
#ymin = trace_summary.loc['E','mean']-trace_summary.loc['E','sd']
#xsd  = trace_summary.loc['T','sd']*2
#ysd  = trace_summary.loc['E','sd']*2
#rect = patches.Rectangle((xmin, ymin), xsd, ysd, linewidth=2, edgecolor='C0', facecolor='none', zorder=10, label=r'Recharge Zone (1$\sigma$)')
#ax.add_patch(rect)
# Add some 2d KDE for recharge zone
#sns.kdeplot(TT[:],EE[:], cmap='coolwarm', shade=True, shade_lowest=False, zorder=12, alpha=0.5)
sns.kdeplot(TT[:20000],EE[:20000], cmap=cmap0, shade=True, thresh=0.05, zorder=12, alpha=0.6)
# Add in well head loc?
ax.axhline(well_elev[well], color='black', linestyle='--', linewidth=1.0)#, label='Well Elevation')
ax.axhline(2950.0, color='black', linestyle=':', linewidth=1.0)#, label='Top Hillslope Elevation')
# Clean up
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Elevation (m)')
ax.yaxis.set_major_locator(MultipleLocator(250))
ax.yaxis.set_minor_locator(MultipleLocator(50))
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.margins(x=0.01)
ax.set_ylim(2400, 3800)
ax.set_xlim(-3,8)
ax.grid(alpha=0.3)
ax.legend(loc='upper right')
#fig.suptitle(well)
fig.tight_layout()
plt.savefig(os.path.join(os.getcwd(), fdir, 'post_laspe_pred_kde.png'), dpi=300)
#plt.savefig(os.path.join(os.getcwd(), fdir, 'post_laspe_pred.svg'), format='svg')
plt.show()





