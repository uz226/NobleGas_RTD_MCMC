#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:13:30 2021

@author: nthiros
"""

# This script plots wells PLM1, PLM7, and PLM6 together
# Must run noble_gas_mcmc.py first for each well


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

_ls=["-.", ":", "--"]


import pymc3 as mc
import seaborn as sb
import theano
import theano.tensor as tt
#from theano import as_op
from theano.compile.ops import as_op
import os
import arviz as az

import seaborn as sns

from scipy import stats


  


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


def lapse_slope_converter(lapse_slope):
    slope_C_m = 1/lapse_slope
    print ('lapse rate slope = {:.3f} (C/100m)'.format(slope_C_100m*100))
    int_2C = lapse_slope*2
    



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


Run_Calib = False   # False to only make plots
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
# variable map
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

scale_map = {'He':1.e8,
             'Ne':1.e7,
             'Ar':1.e4,
             'Kr':1.e8,
             'Xe':1.e8}



# Update to log-beta for Ae and F
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

        



fdir = './figures'


# Read in the MCMC traces
wells = ['PLM1','PLM7','PLM6']

trace_dir = './traces'

idata_comp = {}
trace_summary_comp = {}

for i in wells:
    idata = az.from_netcdf(os.path.join(trace_dir,i+'_trans.netcdf'))
    
    idata_sum = az.summary(idata, round_to=6)
    
    # add median of posterior to statistics
    
    for j in idata_sum.index:
        med = np.median(np.asarray(idata.posterior[j]).ravel())
        idata_sum.loc[j,'median'] = med
    
    idata_sum.to_csv('ng_opt{}.csv'.format(i))
 
    ## Convert from beta parameters to normal
    ##vrs = list(idata.posterior.keys())
    #vrs = ['Ae','F','E'] 
    #for p in vrs:
    #    idata.posterior[p] = idata.posterior[p] * (par_bnd_[p]['max']-par_bnd_[p]['min'])+par_bnd_[p]['min']
    #    #idata.prior[p] = idata.prior[p] * (par_bnd_[p]['max']-par_bnd_[p]['min'])+par_bnd_[p]['min']
    idata_comp[i] = idata
    trace_summary_comp[i] = idata_sum








#
# Combined Marginal Posterior plots
#

var_names = ['m','b','Ae','F','E','T']
     
# DataFrame with MAP estimates
map_df = pd.DataFrame()


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6,4.5))
fig.subplots_adjust(top=0.90,bottom=0.2,left=0.05,right=0.96,hspace=0.85) 
for i in range(len(var_names)):
    r,c = i//3,i%3
    ax = axes[r,c]
        
    v = var_names[i]

    for j in range(len(wells)):
        w = wells[j]
        
        post  = np.asarray(idata_comp[w].posterior[v]).ravel()
        prior = np.asarray(idata_comp[w].prior[v]).ravel()
        
        # find map -- mode of posterior
        grid, pdf = az.kde(post, bw_fct=1.0)
        _map = grid[pdf.argmax()]
        map_df.loc[w,v] = _map
        map_df.loc[w,'1sd_{}'.format(v)] = post.std()
        
        logscl = False
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if v in ['Ae']:
            logscl = True
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        
        if j==0:
            sns.kdeplot(x=prior, ax=ax, bw_adjust=0.25, color='red', alpha=0.8, linewidth=1.5, linestyle='--', log_scale=logscl, label='Prior')
        sns.kdeplot(x=post, ax=ax,  bw_adjust=0.25, color='C{}'.format(j), fill='C{}'.format(j), 
                    linewidth=1.5, linestyle='-', log_scale=logscl, label=w.split('_')[0])
        # Cleaup
        ax.set_xlabel(var_map[var_names[i]], labelpad=0.5)
        ax.minorticks_on()
        ax.tick_params(axis='x',which='both',bottom=True,pad=0.1, labelrotation=45)
        ax.tick_params(axis='y',which='both',left=False,labelleft=False)
        ax.set_ylabel('')
        
        # Trim out really wide priors -- needs to be done manually
        if v == 'F':
            ax.set_xlim(0.0,1.0)
        if v == 'Ae':
            ax.set_xlim(0.001,0.1)
        if v == 'T':
            ax.set_xlim(-0.01,7.01)
            ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(axis='x')
axes[0,0].legend(loc='upper left', bbox_to_anchor=(-0.05, 1.4), ncol=len(wells)+1, frameon=False, 
                 markerscale=0.25, handletextpad=0.5, columnspacing=2.0)
plt.savefig(os.path.join(os.getcwd(), fdir, 'ng_posterior_comp.png'), dpi=300)
plt.savefig(os.path.join(os.getcwd(), fdir, 'ng_posterior_comp.svg'), format='svg')
plt.show()


map_df.index.name = 'well'
map_df.to_csv('./ng_map_df.csv', header=True, index=True)





#
# Lapse Rate KDE
#
from matplotlib.colors import LinearSegmentedColormap
cmap0 = LinearSegmentedColormap.from_list('', ['lightblue','darkblue'])

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(8,3))
fig.subplots_adjust(left=0.12, right=0.98, top=0.8, bottom=0.2, wspace=0.15)

xx = np.linspace(-4, 15, 500) # these are list of temperatures

ll = ['A','B','C']

for a in range(len(wells)):
    w = wells[a]
    
    trace_summary = trace_summary_comp[w]
    idata = idata_comp[w]
    #----
    # Posterior parameter arrays 
    #m, b, Ae, F, E, T = [trace_summary.loc[p, 'mean'] for p in ['m','b','Ae','F','E','T']]
    # The posterior chains -- Ndraws x Mpars
    p_ens = np.array([np.asarray(idata.posterior[p]).ravel() for p in ['m','b','Ae','F','E','T']]).T
    # Drop parameters below 3% hdi and above 97% hdi
    f1 = np.array([p_ens[:,i] >= trace_summary['hdi_3%'][i] for i in range(len(['m','b','Ae','F','E','T']))]).T
    f2 = np.array([p_ens[:,i] <= trace_summary['hdi_97%'][i] for i in range(len(['m','b','Ae','F','E','T']))]).T
    f = np.concatenate((f1,f2),axis=1).all(axis=1)
    #p_ens = p_ens[f]
    # Random shuffle 
    np.random.shuffle(p_ens) 
    # Parameter ensemble arrays for CE mod
    MM, BB, AA, FF, EE, TT = [p_ens[:,i] for i in range(len(['m','b','Ae','F','E','T']))]
    
    ax = axes[a]
    #
    ax.plot(xx, lapse_slope*xx + lapse_b, c='black', linestyle='--', linewidth=2.0, zorder=5, label='Prior Model')
    # maximum a posteriori
    grid, pdf = az.kde(np.asarray(idata['posterior']['m']).ravel(), bw_fct=1.0)
    m_map = grid[pdf.argmax()]
    grid, pdf = az.kde(np.asarray(idata['posterior']['b']).ravel(), bw_fct=1.0)
    b_map = grid[pdf.argmax()]
    
    ax.plot(xx, m_map*xx+b_map, linewidth=3, color='black', zorder=6, label='Max a Posteriori')
    #ax.plot(xx, np.median(np.asarray(idata['posterior']['m']).ravel())*xx + np.median(np.asarray(idata['posterior']['b']).ravel()), 
    #        linewidth=3, color='red', zorder=6, label='Max a Posteriori')
    # error ensembles
    for i in range(500):
        ax.plot(xx, MM[i]*xx + BB[i], c='grey', alpha=0.4)#, label='Posterior Uncertainty' if i==0 else '')
    ax.plot(xx, m_map*xx+b_map, linewidth=1.5, color='grey', alpha=0.75, zorder=4, label='Posterior Uncertainty') # just for legend to show up
    # Add inferred recharge zone
    sns.kdeplot(x=TT[:10000],y=EE[:10000], ax=ax, cmap=cmap0, shade=True, thresh=0.05, zorder=12, alpha=0.6)#, label='Posterior Dist.')
    ax.scatter(-9999,-9999, color='cornflowerblue', alpha=0.7, marker='s', s=80, label='Posterior Distribution')
    # Add in well head loc?
    ax.axhline(well_elev[well], color='black', linestyle=':', linewidth=1.0)#, label='Well Elevation')
    #ax.axhline(2950.0, color='black', linestyle=':', linewidth=1.0)#, label='Top Hillslope Elevation')
    # Clean up
    #ax.set_xlabel('Temperature (C)')
    ax.set_ylabel('Elevation (m)')
    ax.yaxis.set_major_locator(MultipleLocator(200))
    ax.yaxis.set_minor_locator(MultipleLocator(100))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.margins(x=0.01)
    ax.set_ylim(2600, 3500)
    ax.set_xlim(-1,8)
    ax.grid(alpha=0.3)
    #
    if a in [1,2]:
        ax.set_ylabel('')
        ax.tick_params(axis='y',labelleft=False)
    #
    #ax.text(0.5, 0.9, '({})-{}'.format(ll[a], w.split('_')[0]), horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
    ax.text(0.5, 0.9, '{}'.format(w.split('_')[0]), horizontalalignment='center',  verticalalignment='center', transform=ax.transAxes)
axes[1].set_xlabel('Recharge Temperature (C)')
axes[0].legend(ncol=2, loc='lower left', bbox_to_anchor=(0.5, 0.97), frameon=False, 
               handlelength=1.5, labelspacing=0.25, handletextpad=0.5, columnspacing=2.0)
#fig.suptitle(well)
#fig.tight_layout()
plt.savefig(os.path.join(os.getcwd(), fdir, 'post_laspe_pred.png'), dpi=300)
plt.savefig(os.path.join(os.getcwd(), fdir, 'post_laspe_pred.svg'), format='svg')
plt.show()














#--------------------------------------------
#
# Posterior Predictive Concentrations
#
#-----------------------------------------------

ll = ['A','B','C']


gases_ = ['Ne', 'Ar', 'Kr', 'Xe', 'He']


fig, axes = plt.subplots(3,5, figsize=(7.5, 5.5))
fig.subplots_adjust(top=0.96, bottom=0.06, right=0.97, left=0.1, hspace=0.5, wspace=0.6)

for a in range(len(wells)):
    w = wells[a]
    
    trace_summary = trace_summary_comp[w]
    idata = idata_comp[w]
    obs_dict = obs_dict_all[w]

    #----
    # Observations
    obs_arr =  np.array([obs_dict[i] for i in gases_])
    obs_err = obs_arr * np.array([err_dict[i]/100 for i in gases_])
    
    #----
    # Posterior parameter arrays 
    m, b, Ae, F, E, T = [trace_summary.loc[p, 'mean'] for p in ['m','b','Ae','F','E','T']]
    # The posterior chains -- Ndraws x Mpars
    p_ens = np.array([np.asarray(idata.posterior[p]).ravel() for p in ['m','b','Ae','F','E','T']]).T
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

    for i in range(len(gases_)):
        ax = axes[a,i]
        
        gg  = gases_[i]
        scl = scale_map[gg]
        # Predictions using boxplots
        sim_ = C_unc[:,i]*scl
        hdi = az.hdi(sim_, hdi_prob=0.999)
        sim_ = sim_[np.where((sim_>hdi[0]) & (sim_<hdi[1]),True,False)]
        
        kde = sns.kdeplot(y=sim_, ax=ax, shade=True, color='C0', fill='C0', linewidth=1.5, bw_adjust=0.25, label=r'C$_{\mathrm{sim}}$')
        # Observations
        obs_ = np.random.normal(obs_arr[i]*scl, obs_err[i]*scl/1.96, len(sim_))
        sns.kdeplot(y=obs_, ax=ax, color='grey', linewidth=1.5, linestyle='--', bw_adjust=0.25, label=r'C$_{\mathrm{obs}}$')
        # Clean up
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel(r'{} [10$^{{{}}}$]'.format(gases_[i], int(-np.log10(scl))))
        ax.xaxis.labelpad = 5
        ax.yaxis.labelpad = 5
        ax.minorticks_on()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Terrigenic He
        if gg == 'He':
            sns.kdeplot(y=(obs_-sim_), ax=ax, color='C1', fill='C1', linewidth=1.5, linestyle=':', bw_adjust=0.25, label=r'C$_{\mathrm{ter}}$')
            if obs_.max() >= 10.0:
                ax.yaxis.set_major_locator(MultipleLocator(2))
                ax.yaxis.set_minor_locator(MultipleLocator(1))
            if obs_.max() >= 20.0:
                ax.yaxis.set_major_locator(MultipleLocator(5))
                ax.yaxis.set_minor_locator(MultipleLocator(1))
            if obs_.max() >= 50.0:
                print (obs_.max())
                ax.yaxis.set_major_locator(MultipleLocator(10))
                ax.yaxis.set_minor_locator(MultipleLocator(2))
            if obs_.max() < 10.0:
                ax.yaxis.set_major_locator(MultipleLocator(1))
                ax.yaxis.set_minor_locator(MultipleLocator(0.25))
            # For manuscript...
            ax.set_ylim(0,17)
            ax.yaxis.set_major_locator(MultipleLocator(4))
            ax.yaxis.set_minor_locator(MultipleLocator(1))
            
            
            ax.set_xlim(0, ax.get_xlim()[1]*0.5)
            
        ax.text(0.05, 0.92, '({}{})'.format(ll[a],i+1), horizontalalignment='left',  verticalalignment='bottom', transform=ax.transAxes,
                weight='bold', fontsize=14)
            
axes[1,0].set_ylabel(r'Concentration (cm$^{3}$STP/g)')
#fig.suptitle('{} Posterior Predictive'.format(well))
#ax[0].legend()
#fig.tight_layout()
plt.savefig(os.path.join(os.getcwd(), fdir, 'post_conc_pred_kde.png'), dpi=300)
plt.savefig(os.path.join(os.getcwd(), fdir, 'post_conc_pred_kde.svg'), format='svg')
plt.show()








#--------------------------------------------
# Delta Ne
# see kifper 2002 pg. 637 eq. 15
#-----------------------------------------------

gases_ = ['Ne']

dNe_df = pd.DataFrame()

fig, axes = plt.subplots(1,1, figsize=(3,3))
#fig.subplots_adjust()
for a in range(len(wells)):
    w = wells[a]
    ax = axes
    gg  = gases_[0]
    
    trace_summary = trace_summary_comp[w]
    idata = idata_comp[w]
    obs_dict = obs_dict_all[w]

    #----
    # Observations
    obs_arr =  np.array([obs_dict[i] for i in gases_])
    obs_err = obs_arr * np.array([err_dict[i]/100 for i in gases_])
    
    #----
    # Posterior parameter arrays 
    m, b, Ae, F, E, T = [trace_summary.loc[p, 'mean'] for p in ['m','b','Ae','F','E','T']]
    # The posterior chains -- Ndraws x Mpars
    p_ens = np.array([np.asarray(idata.posterior[p]).ravel() for p in ['m','b','Ae','F','E','T']]).T
    # Random shuffle 
    np.random.shuffle(p_ens) 
    # Parameter ensemble arrays for CE mod
    MM, BB, AA, FF, EE, TT = [p_ens[:,i] for i in range(len(['m','b','Ae','F','E','T']))]
    #----
    # Forward predictions of concentrations
    # Maximum a posteriori concentrations
    C_max_ = ng.noble_gas_fun(gases=gases_, E=E, T=T, Ae=Ae, F=F, P='lapse_rate').equil_conc_dry()
    C_max  = np.array([C_max_[i] for i in gases_])
    # Ensemble of predictions for uncertainty
    C_unc = []
    for i in range(20000):
        c = ng.noble_gas_fun(gases=gases_, E=EE[i], T=TT[i], Ae=AA[i], F=FF[i], P='lapse_rate').equil_conc_dry()
        cc = np.array([c[i] for i in gases_])
        C_unc.append(cc)
    C_unc = np.array(C_unc).ravel()

    # Observations
    obs_ = np.random.normal(obs_arr, obs_err/1.96, len(C_unc))
    # delta Ne
    dNe = (np.divide(obs_,C_unc)-1)*100
    kde = sns.kdeplot(y=dNe, ax=ax, color='C{}'.format(a), linewidth=1.5, linestyle='-', bw_adjust=0.25, label=w)
    print ('{} {:.2f} delta Ne'.format(w, dNe.mean()))
    dNe_df.loc[w,'dNe'] = np.median(dNe)
    dNe_df.loc[w,'dNe_1sd'] = dNe.std()
    
    # Clean up
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel(r'$\Delta$Ne (%)')
    ax.minorticks_on()
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 5
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
            
    #ax.text(0.001, dNe.mean(), w, horizontalalignment='left',  verticalalignment='center')#, transform=ax.transAxes, weight='bold', fontsize=14)
    ax.text(kde.get_xlim()[1], dNe.mean(), w, horizontalalignment='left',  verticalalignment='center')            

#axes[1,0].set_ylabel(r'Concentration (cm$^{3}$STP/g)')
#fig.suptitle('{} Posterior Predictive'.format(well))
#axes.legend()
fig.tight_layout()
plt.savefig(os.path.join(os.getcwd(), fdir, 'delNe.png'), dpi=300)
plt.savefig(os.path.join(os.getcwd(), fdir, 'delNe.svg'), format='svg')
plt.show()


dNe_df.index.name = 'well'
dNe_df.to_csv('dNe_comp.csv', index=True, header=True)

