# Script to setup files for age dating MCMC analysis using convolution integral.
# Reads in field obsevations of environmental tracers.
# and propogates noble gas posteriors uncertainties to the field observations.
# 
# Produces the input concentration series for convolution integral.
# Generates posterior predictive distribtion plots for age tracers.
#
# Run noble_gas_mcmc.py before this script
# Actual MCMC is done in age_ens_runs_mcmc directory



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


sys.path.insert(0, './utils')
import noble_gas_utils as ng_utils
import convolution_integral_utils as conv
import cfc_utils as cfc_utils

#import pymc3 as mc
#import theano
#import theano.tensor as TT
##from theano import as_op
#from theano.compile.ops import as_op
import arviz as az

import copy
import seaborn as sns
from itertools import combinations

pd.set_option("display.precision", 8)







#-------------------------------------------------------------------
#
# Field Observation Data Imports
#
#-------------------------------------------------------------------
#
# CFC
cfc_obs = pd.read_excel('./Field_Data/PLM_tracers_2021.xlsx', index_col='Sample', usecols=['Sample','CFC11','CFC12','CFC113'])
cfc_obs += 1.e-10 # avoids zeros for some numerical lubricant

# SF6
sf6_obs = pd.read_excel('./Field_Data/PLM_tracers_2021.xlsx', index_col='Sample', usecols=['Sample','SF6'])
sf6_obs += 1.e-10

# Tritium
h3_obs = pd.read_excel('./Field_Data/PLM_tracers_2021.xlsx', index_col='Sample', usecols=['Sample','H3'])
h3_obs += 1.e-10

# Helium (both 4He and 3He)
he_obs_ = pd.read_excel('./Field_Data/PLM_noblegas_2021.xlsx', skiprows=1, index_col='SiteID', nrows=9)
he_obs  = he_obs_.copy()[['4He','3He']]
he_obs.rename(columns={'4He':'He4','3He':'He3'}, inplace=True)
he_obs.dropna(inplace=True)


#
#----
# List of observations to consider
obs_list = ['PLM1','PLM7','PLM6']
# List of tracers that will be considered
tr_list  = ['CFC11', 'CFC12', 'CFC113', 'SF6', 'He4_ter', 'He3_H3', 'H3']


#
# Uncertainties
# Tracer analytical percent errors / 100
# Assuming all error distributions are normal
# These are 1 sigma
err_dict = {'CFC11':  0.05,
            'CFC12':  0.05,
            'CFC113': 0.05,
            'CFC':    0.05,
            'SF6':    0.05,
            'H3':     0.08, 
            'He4':    0.02,
            'R':      0.015,
            'He3':    0.03} 





#-------------------------------------------------------------------
#
# CE Model Parameters Imports -- Full Posteriors
#
#-------------------------------------------------------------------
# mcmc parameter posteriors produced by noble_gas_mcmc.py in ng_interp dir
np.random.seed(10)

idata = {}
trace_summary = {}
par_map = {}
par_ens = {}

gases = ['He', 'Ne', 'Ar', 'Kr', 'Xe']
pars  = ['m','b','Ae','F','E','T']

par_msk = dict(zip(pars, np.arange(len(pars)))) # dictionary to make sure I am pulling write parameter column

for w in obs_list:
    idata_ = az.from_netcdf('./ng_interp/traces/{}_trans.netcdf'.format(w))
    idata[w] = idata_
    
    trace_summary_ = az.summary(idata_, round_to=6)
    trace_summary[w] = trace_summary_
    #----
    # Posterior parameter arrays 
    par_map[w] = [trace_summary_.loc[p, 'mean'] for p in pars]
    # The posterior chains -- Ndraws x Mpars
    p_ens = np.array([np.asarray(idata_.posterior[p]).ravel() for p in pars]).T
    #----
    #Drop parameters below 3% hdi and above 97% hdi
    #f1 = np.array([p_ens[:,i] >= trace_summary_['hdi_3%'][i] for i in range(len(pars))]).T
    #f2 = np.array([p_ens[:,i] <= trace_summary_['hdi_97%'][i] for i in range(len(pars))]).T
    #p_ens = p_ens[np.concatenate((f1,f2),axis=1).all(axis=1)]
    np.random.shuffle(p_ens) # Random shuffle
    p_ens = p_ens[:50000, :] # Clip number of samples so runs faster
    par_ens[w] = np.array([p_ens[:,i] for i in range(len(pars))]).T
par_ce = ['Ae', 'F', 'E', 'T']
p_inds = [par_msk[p] for p in par_ce]









#-------------------------------------------------------------------
#
# Tracer Atmospheric Input Histories
#
#-------------------------------------------------------------------
#
# CFC and SF6 atm mixing ratios in pptv
# Tritium in TU
# 4He_ter in ccSTP/g
# This is the compiled data from gen_cfc_sf6_ts.py in NOAA_cfc_sf6 dir
# Put everything in tau (years), with tau=0 years at 2021

tr_atm_ = pd.read_csv('./Input_Series/CB_tracer_atmos.csv')
# Extend series back in time
yrs_back = 25000 #50000 -- going back too far slows the convolution integral way down...
tr_atm_arr  = np.row_stack(([tr_atm_.iloc[0,:]]*yrs_back, tr_atm_))
tr_atm_ext  = pd.DataFrame(data=tr_atm_arr, columns=tr_atm_.columns)
# Clean up
tr_atm_ext.index = np.flip(tr_atm_ext.index)
tr_atm_ext.index.name = 'tau'
tr_atm_ext.drop(columns=['Date'], inplace=True)
tr_atm_ext = tr_atm_ext.astype(float)

#
# CFC
cfc_atm = tr_atm_ext.copy()[['CFC12','CFC11','CFC113']]
cfc_atm += 1.e-10
# 
# SF6
sf6_atm = tr_atm_ext.copy()[['SF6']]
sf6_atm += 1.e-10
# 
# Tritium
trit_atm = tr_atm_ext.copy()[['H3']]
trit_atm.columns = ['H3_tu']
# 
# 4He Terrigenic
# The convolution integral script can accumulate terr 4He as function of tau and J 
He4 = np.zeros(len(cfc_atm))
He4_in = pd.DataFrame(index=np.flipud(np.arange(len(He4))))
He4_in['He4_ter'] = He4
# He Production Rates
# Note this can be treated as uncertain in convolution integral -- ie. marginalize over J
#J_ = 4.0/1000/1000*1.e-6 # cc/g/yr  Assumes gam = 1.0, and average upper crust compostition
#J = ng_utils.J_flux(Del=1., rho_r=2700, rho_w=1000, U=3.0, Th=10.0, phi=0.05)
#He4_in_ = np.arange(len(He4))
#He4_in_ter = He4_in_ * J


#
# An input function dictionary keyed by well name then tracer type
CFC11_in, CFC12_in, CFC113_in = cfc_atm[['CFC11']], cfc_atm[['CFC12']], cfc_atm[['CFC113']]
SF6_in = sf6_atm.copy()

C_in_dict = {}
C_in_dict['CFC11']    = CFC11_in
C_in_dict['CFC12']    = CFC12_in
C_in_dict['CFC113']   = CFC113_in
C_in_dict['SF6']      = SF6_in
C_in_dict['He4_ter']  = He4_in
C_in_dict['H3']       = trit_atm

# Save them
with open('age_ens_runs_mcmc/C_in_dict.pk', 'wb') as f:
    pickle.dump(C_in_dict, f)
with open('./C_in_dict.pk', 'wb') as f:
    pickle.dump(C_in_dict, f)

C_in_df = pd.concat((CFC11_in,CFC12_in,CFC113_in,SF6_in,He4_in,trit_atm), axis=1)
C_in_df.index.name = 'tau'
C_in_df.to_csv('C_in_df.csv', index=True)



#-------------------------------------------------------------------
#
# Convert Field Observations to pptv
#
#-------------------------------------------------------------------
#
# Correct aqeuous field observations of CFC and SF6 for
# excess-air and recharge temperature/pressure.
# Put in units of PPTV
#
# Update 1/17/2022: Include analytical uncertainties for the ensemble

#
# CFC
#
cfc_wells = ['PLM1','PLM6','PLM7']

# Convert field aqeous concentration to an atm mixing ratios
cfc_ppt_map = {} # using the CE model max. a posteriori parameter set
cfc_ppt_ens = {} # using the CE model posterior distributions
for w in cfc_wells:
    # aqueous field obs in pmol/kg
    cmeas_aq  = cfc_obs.loc[w, ['CFC11','CFC12','CFC113']].to_numpy()   
    if w not in list(par_map.keys()):
        print ('cannot find {} in par_map dictionary'.format(w))
    if w not in list(par_ens.keys()):
        print ('cannot find {} in par_ens dictionary'.format(w))
    # Max a Posteriori parameter set
    Ae, F, E, T = np.array(par_map[w])[p_inds]
    cfc_ppt_map[w] = cfc_utils.cfc_ce_corr(cfc_num=[11,12,113],E=E,T=T,Ae=Ae,F=F).equil_air_conc_cfc(cmeas_aq)
    #
    # Array of analytical uncertainties
    unc_ = np.array([np.random.normal(0.0, cmeas_aq[i]*err_dict['CFC'], len(par_ens[w])) for i in range(len(cmeas_aq))]).T
    # Ensemble using full NG posteriors
    ens_pptv = []
    for i in range(len(par_ens[w])):
        # Parameters
        Ae, F, E, T = par_ens[w][i, p_inds]
        # Realization of obs including unc
        cmeas_aq_unc = cmeas_aq.copy() + unc_[i,:]
        # Convert aqeous conc. to atm mixing ratio in pptv
        #ens_pptv.append(cfc_utils.cfc_ce_corr(cfc_num=[11,12,113],E=E,T=T,Ae=Ae,F=F).equil_air_conc_cfc(cmeas_aq))
        ens_pptv.append(cfc_utils.cfc_ce_corr(cfc_num=[11,12,113],E=E,T=T,Ae=Ae,F=F).equil_air_conc_cfc(cmeas_aq_unc))
    cfc_ppt_ens[w] = pd.DataFrame(ens_pptv, columns=['CFC11','CFC12','CFC113'])
cfc_ppt_map = pd.DataFrame(cfc_ppt_map).T
cfc_ppt_map.columns = ['CFC11','CFC12','CFC113']


#
# SF6
#
sf6_wells = ['PLM1','PLM6','PLM7']
# Convert field aqeous concentration to an atm mixing ratios
sf6_ppt_map = {} # using the CE model max. a posteriori parameter set
sf6_ppt_ens = {} # using the CE model posterior distributions
for w in sf6_wells:
    # aqueous field obs in pmol/kg
    cmeas_aq  = sf6_obs.loc[w, 'SF6']
    if w not in list(par_map.keys()):
        print ('cannot find {} in par_map dictionary'.format(w))
    if w not in list(par_ens.keys()):
        print ('cannot find {} in par_ens dictionary'.format(w))
    # Max a Posteriori
    Ae, F, E, T = np.array(par_map[w])[p_inds]
    sf6_ppt_map[w] = cfc_utils.sf6_ce_corr(E=E,T=T,Ae=Ae,F=F).equil_air_conc_sf6(cmeas_aq)
    #
    # Array of analytical uncertainties
    unc_ = np.random.normal(0.0, cmeas_aq*err_dict['SF6'], len(par_ens[w]))
    # Ensemble
    ens_pptv = []
    for i in range(len(par_ens[w])):
        Ae, F, E, T = par_ens[w][i, p_inds]
        cmeas_aq_unc = cmeas_aq.copy() + unc_[i]
        # Convert aqeous conc. to atm mixing ratio in pptv
        #ens_pptv.append(cfc_utils.sf6_ce_corr(E=E,T=T,Ae=Ae,F=F).equil_air_conc_sf6(cmeas_aq))
        ens_pptv.append(cfc_utils.sf6_ce_corr(E=E,T=T,Ae=Ae,F=F).equil_air_conc_sf6(cmeas_aq_unc))
    sf6_ppt_ens[w] = pd.DataFrame(np.array([ens_pptv]).T, columns=['SF6'])
sf6_ppt_map = pd.DataFrame([sf6_ppt_map]).T
sf6_ppt_map.columns = ['SF6']



#
# Helium
#
par_ce = ['Ae', 'F', 'E', 'T']
# Modeling Rter
#Rterr = 2.0e-8 -- literature value
# Log-nomral Rterr
#mu  = np.log10(2.0e-8) # prior mean based on Kipfer 2002
#std = 0.25
#std = 0.05
#Rterr_rv_log = np.random.normal(abs(mu), abs(std), len(par_ens['PLM7']))
#Rterr_rv = 10**-Rterr_rv_log
#
# Beta Rterr in log space
low_  = np.log10(2.e-8) - 1.0
high_ = np.log10(2.e-8) + 1.0
beta_ = np.random.beta(2, 2, len(par_ens['PLM7']))
Rterr_rv = 10**(beta_ * (high_-low_)+low_)
# Normal distribution
#Rterr_rv = np.random.normal(2.e-8, 0.2*2.e-8, len(par_ens['PLM7']))
#Rterr = Rterr_rv.mean()
#
Rterr   = 10**(np.log10(Rterr_rv).mean()) # use this if log distribution for rterr
Rterr_c = np.ones_like(Rterr_rv)*Rterr    # make it a list of constant values


#
# MAP -- uses map T,E,Ae,F and a constant a-priori Rterr for 3He
# Does not include analytical uncertainty in 4He_obs or 3He_obs
He_map = {}
for w in list(par_map.keys()):
    obs_w = copy.deepcopy(he_obs).T.to_dict()[w]
    Ae, F, E, T = [par_map[w][par_msk[p]] for p in par_ce]
    # Calculate He comps 
    ngp = ng_utils.ng_parse(obs_w, Ae, F, E, T)
    ngp.He_comps(Rterr)
    He_map[w] = copy.deepcopy(ngp.obs_dict_)
he_map       = pd.DataFrame(He_map).T
he4_atm_map  = pd.DataFrame([he_map['He4_atm']]).T  # sol. eq. + excess air
he4_eq_map   = pd.DataFrame([he_map['He4_eq']]).T   # sol. eq.
he4_ter_map  = pd.DataFrame([he_map['He4_ter']]).T  # terrigenic
he3_map      = pd.DataFrame([he_map['He3_tu']]).T   # tritiogenic 3He


#
# Generate Ensemble of Helium concentration as function of Rterr
# Includes 4He and 3He obs percent errors (01/17/22)
def sample_he_ens(R_terr_list):
    # Helium ensembles -- uses full posterior of T, E, Ae, F 
    # and a constant a-priori Rterr for 3He
    he4_ens_     = {} # terrigenic
    he4_ens_eq_  = {} # solubility equilibrium 
    he4_ens_atm_ = {} # solubility equilibrium + ex. air
    he4_del_     = {} # delta helium
    he3_ens_     = {} # tritogenic
    p_inds = [par_msk[p] for p in par_ce]
    for w in list(par_ens.keys()): # sample over all wells
        obs_w = copy.deepcopy(he_obs).T.to_dict()[w]
        ens_pred_ = []
        for i in range(len(par_ens[w])): # sample over CE-model parameters
            # Randomly sample over analytical uncertainties
            obs_w_unc = copy.deepcopy(obs_w)
            obs_w_unc['He4'] += np.random.normal(0, obs_w['He4']*err_dict['He4'])
            obs_w_unc['He3'] += np.random.normal(0, obs_w['He3']*err_dict['He3'])
            Ae, F, E, T = par_ens[w][i, p_inds]
            ngp = ng_utils.ng_parse(obs_w_unc, Ae, F, E, T)
            ngp.He_comps(R_terr_list[i])
            ens_pred_.append(copy.deepcopy(ngp.obs_dict_))
        he4_ens_[w]      = pd.DataFrame(ens_pred_)[['He4_ter']]
        he4_ens_eq_[w]   = pd.DataFrame(ens_pred_)[['He4_eq']]
        he4_ens_atm_[w]  = pd.DataFrame(ens_pred_)[['He4_atm']]
        he3_ens_[w]      = pd.DataFrame(ens_pred_)[['He3_tu']]
        he4_del_[w]      = pd.DataFrame(ens_pred_)[['He4_del']]
    return copy.deepcopy(he4_ens_), copy.deepcopy(he4_ens_eq_), copy.deepcopy(he4_ens_atm_), copy.deepcopy(he3_ens_), copy.deepcopy(he4_del_)
    
he4_ens, he4_ens_eq, he4_ens_atm, he3_ens_cr, he4_del_cr =  sample_he_ens(Rterr_c)   # constant Rterr
he4_ens, he4_ens_eq, he4_ens_atm, he3_ens, he4_del       =  sample_he_ens(Rterr_rv)  # variable Rterr

#
# Marginalize over Rterr -- calculate many posterior predictive distributions assuming constant Rterr
Rterr_unifrom_ = np.linspace(np.log10(Rterr_rv.min()), np.log10(Rterr_rv.max()), 5)
# Add a very small Rterr to make 4He_ter*Rter negligable. Should result in youngest possible age (small 3H/He3 ratio)
Rterr_unifrom_ = np.concatenate(([-12], Rterr_unifrom_))
he3_ens_marg = {}
for rl in Rterr_unifrom_:
    r = 10**rl
    rt = np.ones_like(Rterr_rv)*(r)
    _,_,_,he3_ens_m,_ = sample_he_ens(rt) 
    he3_ens_marg[-rl] = he3_ens_m


#
# 3H/3He ratios
# MAP E,T,Ae,F and constant Rterr
h3_he3_map = he3_map.copy()
for w in h3_he3_map.index:
    try:
        h3_he3_map.loc[w,'H3_He3'] = h3_obs.loc[w, 'H3'] / h3_he3_map.loc[w,'He3_tu']
    except KeyError:
        pass
h3_he3_map = h3_he3_map[['H3_He3']]


# 3H Concentration w/ analytical uncertainty
# 3H_initial concentration
# 3H/3He Ratios Ensemble
# Includes uncertainty analytical 3H uncertaintly
# and 3He_tu uncertainty as function of: T, E, Ae, F, Rterr and analytical observations of He isotopes
h3_he3_ens_cr = {}  # constant Rterr
h3_he3_ens    = {}  # variable Rterr
h3_init_ens   = {}
h3_ens        = {}
for w in list(he3_ens.keys()): # sample over each well
    try:
        he3_cr = he3_ens_cr[w].copy() # constant rterr, already includes a bunch of uncertainties (see above)
        he3    = he3_ens[w].copy()    # variable rterr
        h3     = np.random.normal(h3_obs.loc[w,'H3'], h3_obs.loc[w,'H3']*err_dict['H3'], len(he3)) # tritium obs + analytical error
        ## Constant Rterr
        d = h3[:,np.newaxis] / he3_cr
        d.columns = ['H3_He3']
        h3_he3_ens_cr[w] = d
        ## Variable Rterr
        d = h3[:,np.newaxis] / he3  # h3_obs.loc[w,'3H'] / he3 
        d.columns = ['H3_He3']
        h3_he3_ens[w] = d
        ## Initial Tritium
        h3_ens[w]      = pd.DataFrame(h3, columns=['H3'])
        h3_init_ens[w] = pd.DataFrame(he3.to_numpy().ravel() + h3, columns=['H3_init'])
    except KeyError:
        pass

    





#-------------------------------------------------------------------
#
# Observation Data Clean-up
#
#-------------------------------------------------------------------
#
# Combined MAP tracer dictionary -- includes MAP T, E, Ae, and F
map_dict = {}
map_dict['CFC']     = copy.deepcopy(cfc_ppt_map)
map_dict['SF6']     = copy.deepcopy(sf6_ppt_map)
map_dict['He4_ter'] = copy.deepcopy(he4_ter_map)
map_dict['H3_He3']  = copy.deepcopy(h3_he3_map)  # a-priori Rterr
map_dict['He3']     = copy.deepcopy(he3_map.rename(columns={'He3_tu':'He3'}))     # variable Rterr
map_dict['H3']      = copy.deepcopy(h3_obs.rename(columns={'3H':'H3'}))


# Expand ens dict into individual CFC to speed things up
for i in ['CFC11','CFC12','CFC113']:
    #map_dict[i] = {}
    d = pd.DataFrame([map_dict['CFC'][i]]).T
    map_dict[i] = d


# Combined ensemble tracer dictionary -- includes uncertain T, E, Ae, and F
# and analytical uncertainties
ens_dict = {}
ens_dict['CFC']         = copy.deepcopy(cfc_ppt_ens)
ens_dict['SF6']         = copy.deepcopy(sf6_ppt_ens)
ens_dict['He4_ter']     = copy.deepcopy(he4_ens)
ens_dict['He4_ter_del'] = copy.deepcopy(he4_del)
ens_dict['H3_He3']      = copy.deepcopy(h3_he3_ens)  # variable Rterr
ens_dict['He3']         = copy.deepcopy(he3_ens)
ens_dict['H3']          = copy.deepcopy(h3_ens)
ens_dict['H3_init']     = copy.deepcopy(h3_init_ens)


# Expand ens dict into individual CFC to speed things up
for i in ['CFC11','CFC12','CFC113']:
    ens_dict[i] = {}
    for w in list(ens_dict['CFC'].keys()):
        d = pd.DataFrame([copy.deepcopy(ens_dict)['CFC'][w][i]]).T
        ens_dict[i][w] = d

        
# Save them
with open('age_ens_runs_mcmc/map_dict.pk', 'wb') as f:
    pickle.dump(map_dict, f)
with open('./map_dict.pk', 'wb') as f:
    pickle.dump(map_dict, f)


with open('age_ens_runs_mcmc/ens_dict.pk', 'wb') as f:
    pickle.dump(ens_dict, f)
with open('./ens_dict.pk', 'wb') as f:
    pickle.dump(ens_dict, f)


# quick data frame with MAPS and std
c_df = pd.DataFrame()
for t in ['He4_ter','He4_ter_del','CFC12','SF6','H3']:
    for w in ['PLM1','PLM7','PLM6']:
        _cc = ens_dict[t][w]
        c_df.loc[w,t] = float(_cc.mean())
        c_df.loc[w,'1sd_{}'.format(t)] = float(_cc.std())
c_df.index.name='well'
c_df.to_csv('posterior_predictive_conc.csv',index=True, header=True)








#--------------------------------------------
#
# Plotting
#
#--------------------------------------------

fdir = './figures'


#-------------------------------------------
#
# Concentration Posterior Predictive  Plots
#
#-------------------------------------------

# want to compare the posterior distributions of the concentrations
# against an analytical errors.

# Plotting label utils
ylabs = {'CFC11':'CFC-11',
         'CFC12':'CFC-12',
         'CFC113':'CFC-113',
         'SF6':'SF$_{6}$',
         'He4_ter':'$^{4}$He$\mathrm{_{terr}}$',
         'H3_He3':'$\mathrm{^{3}H/^{3}He}$',
         'H3':'$\mathrm{^{3}H}$'}

ylabs_units = {'CFC11':'pptv',
               'CFC12':'pptv',
               'CFC113':'pptv',
               'SF6':'pptv',
               'He4_ter':r'cm$^{3}$STP/g',
               'H3_He3':'TU',
               'H3':'TU'}

 

#
# Plot ensemble distribution versus analytical error alone
#
wells    = ['PLM1','PLM7','PLM6']
tracers  = ['CFC11','CFC12','SF6','H3_He3','He4_ter']

fig, axes = plt.subplots(3,5, figsize=(9,7))
fig.subplots_adjust(top=0.96, bottom=0.18, left=0.05, right=0.96, wspace=0.1, hspace=0.45)
for i in range(15):
    r,c = i//5, i%5    
    w = wells[r]
    t = tracers[c]
    ax = axes[r,c]
    
    # Analytical Error Estmiate
    try:
        perr   = err_dict[t]
    except KeyError:
        perr   = 0.015
    munc_mu = ens_dict[t][w].to_numpy().ravel().mean()
    munc_sd = munc_mu * perr
    munc  = np.random.normal(munc_mu, munc_sd, len(ens_dict[t][w].to_numpy().ravel()))
    if t == 'He4_ter':
        sns.kdeplot(x=munc*1.e8, color='grey', linestyle='--', alpha=0.5, ax=ax, bw_adjust=0.8)
    else:
        sns.kdeplot(x=munc, color='grey', linestyle='--', alpha=0.5, ax=ax, bw_adjust=0.8)
    if r == 0:
        ax.text(0.65, 0.8, '{:.1f}%'.format(perr*100), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
    # Full ensemble Posterior Predictive
    if t == 'He4_ter':
        ens_pred = ens_dict[t][w].to_numpy().ravel()*1.e8
    else: 
        ens_pred = ens_dict[t][w].to_numpy().ravel()
    sns.kdeplot(x=ens_pred, color='black', ax=ax, bw_adjust=0.5)
    # Clean-up
    ax.tick_params(axis='y',which='both',labelleft=False,left=False)
    ax.tick_params(axis='x',rotation=45)
    ax.set_ylabel('')
    # Xtick locs
    min_, max_, mean_, sd_  = ens_pred.min(), ens_pred.max(), ens_pred.mean(), 3*ens_pred.std()
    if mean_ - min_ > 1:
        ax.set_xticks(np.array([mean_-sd_, mean_, mean_+sd_]).round(1))
    else:
        ax.set_xticks(np.array([mean_-sd_, mean_, mean_+sd_]).round(2))
    # Labeling
    if c == 0:
        ax.set_ylabel(w.split('_')[0])
    if r == 2:
        if t == 'He4_ter':
            ax.set_xlabel('{}\n(10$^{{-8}}$ {})'.format(ylabs[t], ylabs_units[t]))
        else:
            ax.set_xlabel('{} ({})'.format(ylabs[t], ylabs_units[t]))
#plt.savefig('./figures/concentration_posterior_prd.png',dpi=300)
plt.show()






#-------------------------------------------------
#
# Helium Uncertainty Decomp Plots
#
#-------------------------------------------------
# 3He
#
# Plot Rterr
fig, ax = plt.subplots(figsize=(4,3))
sns.kdeplot(x=Rterr_rv, ax=ax, color='C0', fill=False, linewidth=1.5, linestyle='-', log_scale=True)
#ax.hist(Rterr_rv, bins=np.logspace(np.log10(Rterr_rv.min()),np.log10(Rterr_rv.max()), 50), histtype='step', color='C0', linewidth=1.5, linestyle='-')
#ax.scatter(Rterr_rv.mean(), 0.0, marker='|', s=200, color='C0')
#ax.axvline(Rterr_rv.mean(), color='C0', linestyle='--')
ax.set_xlabel(r'$(^{3}He/^{4}He)_{ter}$')
#
# Plot the marginal values
for rt in list(he3_ens_marg.keys()):
    ax.scatter(10**-rt, 0.01, marker='o', color='grey')
# Cleanup
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_ylabel('')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
#ax.set_xlim(1.e-9, 1.e-7)
#ax.set_xscale('log')
fig.tight_layout()
plt.savefig('figures/Rterr.png', dpi=300)
plt.show()


# Plot Helium-3 comps
wells    = ['PLM1','PLM7','PLM6']
fig, axes = plt.subplots(3,2, figsize=(6,6))
fig.subplots_adjust(top=0.92, bottom=0.11, left=0.1, right=0.96, hspace=0.3)
for i in range(len(wells)):
    w  = wells[i]
    ax = axes[i,0]
    #
    # tritiogenic uncertainty constant Rterr
    he3_tu_c  = copy.deepcopy(he3_ens_cr[w]).to_numpy().ravel()
    # tritiogenic uncertainty variable Rterr
    he3_tu_v  = copy.deepcopy(he3_ens[w]).to_numpy().ravel()
    #
    # Field Obs Uncertainty
    h3_  = h3_obs.loc[w,'H3'].copy()
    h3   = np.random.normal(h3_, h3_*0.1/1.96, len(he3_tu_c)) 
    # Ratio
    rat_c = copy.deepcopy(h3_he3_ens_cr[w]).to_numpy().ravel() # same eq. as h3_he3_ens below
    rat_v = copy.deepcopy(h3_he3_ens[w]).to_numpy().ravel()
    #
    #
    sns.kdeplot(x=h3, ax=ax, color='C2', fill='C2', linewidth=1.5, linestyle=':', zorder=8, label=r'$^{3}H_{obs}$')
    sns.kdeplot(x=he3_tu_c, ax=ax, color='C0', fill='C0', linewidth=1.5, linestyle='-', zorder=8, label=r'$^{3}He^{\star}_{c}$')
    sns.kdeplot(x=he3_tu_v, ax=ax, color='C1', fill='C1', linewidth=1.5, linestyle='--', zorder=8, label=r'$^{3}He^{\star}_{v}$')
    # Cleanup
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.set_ylim(ax.get_ylim()[0],ax.get_ylim()[1]*0.8)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel(w.split('_')[0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #
    #  Ratios
    ax2 = axes[i,1]
    sns.kdeplot(x=rat_c, ax=ax2, color='C0', fill='C0', linewidth=1.5, linestyle='-', zorder=8,label=r'$^{3}H / ^{3}He^{\star}_{c}$')
    sns.kdeplot(x=rat_v, ax=ax2, color='C1', fill='C1', linewidth=1.5, linestyle='--', zorder=8, label=r'$^{3}H / ^{3}He^{\star}_{v}$')
    # Cleanup
    if (rat_v.max()-rat_v.min()) > 0.25:
        ax2.xaxis.set_major_locator(MultipleLocator(0.1))
        ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
    else:
        ax2.xaxis.set_major_locator(MultipleLocator(0.05))
        ax2.xaxis.set_minor_locator(MultipleLocator(0.025))
    #ax2.set_ylim(ax.get_ylim()[0],ax.get_ylim()[1]*1.1)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_ylabel('')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    #
    ## Add in Marginalized stuff
    #for rt in list(he3_ens_marg.keys()):
    #    he3_tu_m = he3_ens_marg[rt][w].to_numpy().ravel()
    #    sns.kdeplot(x=he3_tu_m, ax=ax, color='grey', fill='grey', linewidth=0.5, alpha=0.25, zorder=5, linestyle='-')#, label=r'$^{3}He^{\star}_{c}$')
    #    ratio = h3_obs.loc[w,'3H'] / he3_tu_m
    #    sns.kdeplot(x=ratio, ax=ax2, color='grey', fill='grey', linewidth=0.5, alpha=0.25, zorder=5, linestyle='-')
    
axes[2,0].set_xlabel('Concentration (TU)')
axes[2,1].set_xlabel(r'$^{3}H / ^{3}He^{\star}$')
axes[0,0].legend(bbox_to_anchor=(0.05, 1.4), ncol=1, loc='upper left', frameon=False, labelspacing=0.15)
plt.savefig(os.path.join(os.getcwd(), fdir, 'He3_posterior_prd.png'), dpi=300)
plt.savefig(os.path.join(os.getcwd(), fdir, 'He3_posterior_prd.svg'), format='svg')
plt.show()






#
#-------------------------------------------
#
# Tracer-Tracer Concentration/Mixing Plots
#
#-------------------------------------------
#

# Generate some concentration versus tau
# for piston flow and exponential RTD



tau_list = np.arange(2, 10005, 1)
tau_list = np.concatenate(([1], tau_list))


rerun = True
if rerun: 
    tau_df_exp = pd.DataFrame(index=tau_list) # dataframe to hold simulated concentrations
    tau_df_pf = pd.DataFrame(index=tau_list)  # dataframe to hold simulated concentrations
    
    tlist = ['CFC11', 'CFC12', 'CFC113', 'SF6', 'He4_ter', 'H3', 'He3']
    
    # Forward convolution -- Exponential Model
    #for w in obs_list:
    for t in tlist:
        print ('Working on {}'.format(t))
        # Initialize Models
        if t in ['H3','He3']:
            tr_in = C_in_dict['H3'].copy()
        else:
            tr_in = C_in_dict[t].copy()
        conv_ = conv.tracer_conv_integral(C_t=tr_in, t_samp=tr_in.index[-1])
        for i in tau_list:
            if t == 'H3':
                conv_.update_pars(tau=i, mod_type='exponential', t_half=12.34)
            elif t == 'He3':
                conv_.update_pars(tau=i, mod_type='exponential', t_half=12.34, rad_accum='3He')
            elif t == 'He4_ter':
                J = ng_utils.J_flux(Del=1., rho_r=2700, rho_w=1000, U=3.0, Th=10.0, phi=0.05)
                conv_.update_pars(tau=i, mod_type='exponential', rad_accum='4He',J=J)
            else:
                conv_.update_pars(tau=i, mod_type='exponential')
            # Perform Exponential model    
            c_exp = conv_.convolve()
            tau_df_exp.loc[i, t] = c_exp
            # Update tau for piston-flow model
            conv_.mod_type = 'piston'
            c_pf = conv_.convolve()
            tau_df_pf.loc[i, t] = c_pf
    tau_df_exp['H3_He3'] = tau_df_exp['H3']/tau_df_exp['He3']
    tau_df_pf['H3_He3'] = tau_df_pf['H3']/tau_df_pf['He3']
    # Save it so does not need to be done again
    # This is static, does not depend on parameters
    tau_df_exp.to_csv('tau_df_exp.csv')
    tau_df_pf.to_csv('tau_df_pf.csv')
        
else:
    tau_df_exp = pd.read_csv('tau_df_exp.csv', index_col=0)
    tau_df_pf = pd.read_csv('tau_df_pf.csv', index_col=0)
    
 
    
 
"""
#---------
# Testing influence of background 3H concentrations
tau_df_exp_tutest = pd.DataFrame(index=tau_list) # dataframe to hold simulated concentrations
tlist = ['H3', 'He3']
        
# Forward convolution -- Exponential Model
#for w in obs_list:
for t in tlist:
    print ('Working on {}'.format(t))
    # Initialize Models
    if t in ['H3','He3']:
        tr_in = C_in_dict['H3'].copy()
        tr_in[tr_in==3.327329892219456]=8.0
    else:
        tr_in = C_in_dict[t].copy()
    conv_ = conv.tracer_conv_integral(C_t=tr_in, t_samp=tr_in.index[-1])
    for i in tau_list:
        if t == 'H3':
            conv_.update_pars(tau=i, mod_type='exponential', t_half=12.34)
        elif t == 'He3':
            conv_.update_pars(tau=i, mod_type='exponential', t_half=12.34, rad_accum='3He')
        # Perform Exponential model    
        c_exp = conv_.convolve()
        tau_df_exp_tutest.loc[i, t] = c_exp
tau_df_exp_tutest['H3_He3'] = tau_df_exp_tutest['H3']/tau_df_exp_tutest['He3']
"""
 
    
 

    
#----------------------------------------------------------------
#
# Tracer versus Tau plots
#
#----------------------------------------------------------------    
nruns = len(ens_dict['CFC12'][list(ens_dict['CFC12'].keys())[0]])

def plot_tau_vs_cfc(well_list, tracer_list, savename):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5,7))
    fig.subplots_adjust(top=0.98, bottom=0.1, left=0.18, right=0.65, hspace=0.25)
    for i in range(3):
        ax = axes[i]
        tr = tracer_list[i]
        #smp = '{}_{}'.format(wn, tr)
        ax.semilogx(tau_df_exp[tr], color='black')
        ax.semilogx(tau_df_pf[tr], color='black', linestyle='--')  
        for l in range(len(well_list)):
            wn = well_list[l]
            #ax.set_ylabel('{} {}-{} (pptv)'.format(wn.split('_')[0], tr[:3], tr[3:]))
            #ax.set_ylabel('{} {} ({})'.format(wn.split('_')[0], ylabs[tr], ylabs_units[tr]))
            ax.set_ylabel('{} ({})'.format(ylabs[tr], ylabs_units[tr]))
            # Add field observation
            #ax.axhline(map_dict[tr].loc[wn, tr], color='black', linestyle=':', alpha=0.7)
            # ensemble of corrected field observations
            ens_pred = np.array([ens_dict[tr][wn].loc[j,tr] for j in range(nruns)])
            ax.axhline(ens_pred.mean(), color='C{}'.format(l), linestyle='-', alpha=0.7, label=wn)
            ax.fill_between(tau_list, ens_pred.mean()-ens_pred.std()*3, ens_pred.mean()+ens_pred.std()*3, 
                            color='C{}'.format(l), alpha=0.1) 
        # Clean-up
        ax.set_xlim(1, 10005)
        ax.xaxis.set_major_locator(ticker.LogLocator(numticks=8))
        ax.minorticks_on()
        ax.grid()
        if tr == 'CFC11':
            #ax.set_ylim(-0.5, 30.0)
            #ax.yaxis.set_major_locator(MultipleLocator(10))
            #ax.yaxis.set_minor_locator(MultipleLocator(2))
            pass
        if tr == 'CFC12':
            #ax.set_ylim(-0.5, 100.0)
            #ax.yaxis.set_major_locator(MultipleLocator(25))
            #ax.yaxis.set_minor_locator(MultipleLocator(5))
            pass
        if tr == 'CFC113':
            #ax.set_ylim(-0.25, 2.0)
            #ax.yaxis.set_major_locator(MultipleLocator(1))
            #ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            pass
        if tr == 'SF6':
            #ax.set_ylim(0, 3.0)
            #ax.yaxis.set_major_locator(MultipleLocator(1))
            #ax.yaxis.set_minor_locator(MultipleLocator(0.25))
            pass
        if tr == 'H3_He3':
            #ax.set_ylim(0.0, 1.0)
            #ax.yaxis.set_major_locator(MultipleLocator(5.0))
            #ax.yaxis.set_minor_locator(MultipleLocator(2.5))
            pass
        if tr == 'H3':
            #ax.set_ylim(0.0, 10.0)
            #ax.yaxis.set_major_locator(MultipleLocator(2))
            #ax.yaxis.set_minor_locator(MultipleLocator(1))
            pass
        if tr == 'He4_ter':
            #ax.set_ylim(1.e-9, 1.e-6)
            ax.set_yscale('log')
    # Clean up
    axes[2].set_xlabel('Mean Age (years)')
    axes[0].legend(loc='upper left', bbox_to_anchor=(0.7, 1.1), handlelength=1.0, handletextpad=0.4, framealpha=0.9)
    #axes[0].legend(loc='upper right', handlelength=1.0, handletextpad=0.4)
    if savename:
        plt.savefig(os.path.join(os.getcwd(),fdir,'{}.jpg'.format(savename)), dpi=300)
    plt.show() 


# 
obslist1 = ['PLM1', 'PLM6', 'PLM7']
plot_tau_vs_cfc(obslist1, ['CFC11', 'CFC12','SF6'], 'CFC_vs_tau.plm1')
plot_tau_vs_cfc(obslist1, ['H3', 'H3_He3','He4_ter'], 'C_vs_tau')

    
 
 
    
 
#   
# Tracer-Tracer Plots 
#

# Set up color maps
from matplotlib.colors import LinearSegmentedColormap
min_val, max_val = 0.3,1.0
n = 10
cmlist = []
for i in [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens, plt.cm.Reds]:
    colors = i(np.linspace(min_val, max_val, n))
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)
    cmlist.append(cmap)


#--------------------------
# CFC11 vs CFC12 
#--------------------------
fig, axes = plt.subplots(1,2,figsize=(7,3.5))
fig.subplots_adjust(top=0.88, bottom=0.17, left=0.15, right=0.96, hspace=0.35, wspace=0.2)
wells = ['PLM1', 'PLM7', 'PLM6']
wt1 = 'CFC11'
wt2 = 'CFC12'
for j in range(2):
    ax = axes[j]
    # Labeling of pistonflow line
    tau_df_pf_ = tau_df_pf[tau_df_pf.index <= 76]
    #pf1 = tau_df_pf_.loc[::5, wt1]
    #pf2 = tau_df_pf_.loc[::5, wt2]
    yrs_lab = np.arange(0,71,10)
    yrs_lab[0] = 1
    pf1 = tau_df_pf_.loc[yrs_lab,wt1]
    pf2 = tau_df_pf_.loc[yrs_lab,wt2]
    # Labeling of exponential line
    dates = [1, 10, 100, 500, 1000, 9000]
    exp1 = tau_df_exp.loc[dates, wt1]
    exp2 = tau_df_exp.loc[dates, wt2]
    #--
    # Piston-Flow
    ax.plot(tau_df_pf[wt1], tau_df_pf[wt2], color='grey', zorder=5)
    ax.scatter(pf1, pf2, color='grey', marker='o', facecolors='none', s=8, zorder=5)
    [ax.text(pf1.iloc[i], pf2.iloc[i], pf2.index[i], color='grey', fontsize=12) for i in range(len(pf1))]
    # Exponential
    ax.plot(tau_df_exp[wt1], tau_df_exp[wt2], color='black', linestyle='--', zorder=5)
    ax.scatter(exp1, exp2, color='black', marker='D', facecolors='none', s=8, zorder=5)
    [ax.text(exp1.iloc[i], exp2.iloc[i], exp2.index[i], fontsize=12) for i in range(1,len(exp1))]
    # Field Data
    for i in range(len(wells)):
        w  = wells[i]
        if j == 0:
            #ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
            ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
        elif j == 1:
            xx = ens_dict[wt1][w].to_numpy().ravel()[:20000]
            yy = ens_dict[wt2][w].to_numpy().ravel()[:20000]
            sns.kdeplot(x=xx,y=yy, ax=ax, cmap=cmlist[i], thresh=0.01, zorder=3, shade=True)
            #ax.scatter(ens_dict[wt1][w][:2500], ens_dict[wt2][w][:2500], color='C{}'.format(i), marker='.', alpha=0.5, zorder=3)
        # Clean up
        #ax.set_yscale('log')
        #ax.set_xscale('log') 
    if j==0: 
        ax.set_ylabel('{} ({})'.format(ylabs[wt2], ylabs_units[wt2]))
    ax.set_xlabel('{} ({})'.format(ylabs[wt1], ylabs_units[wt1]))
    ax.set_xlim(-40, 305)
    ax.set_ylim(-90, 605)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    # Zoom in on right for inset
    if j==1:
        ax.set_xlim(-4.0, 26)
        ax.set_ylim(-5.0, 60)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.text(0.95, 0.95, 'Inset', horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='white', linewidth=0.1), zorder=10)
fig.legend(ncol=3,loc='center left', bbox_to_anchor=(0.25, 0.95), frameon=False)
plt.savefig(os.path.join(fdir,'CFC11_CFC12_tracer_comp.svg'), format='svg')
plt.show()


#--------------------------
# CFC113 vs CFC12 
#--------------------------
fig, axes = plt.subplots(1,2,figsize=(7,3.5))
fig.subplots_adjust(top=0.88, bottom=0.17, left=0.15, right=0.96, hspace=0.35, wspace=0.2)
wells = ['PLM1', 'PLM7', 'PLM6']
wt1 = 'CFC113'
wt2 = 'CFC12'
for j in range(2):
    ax = axes[j]
    # Labeling of pistonflow line
    tau_df_pf_ = tau_df_pf[tau_df_pf.index <= 76]
    #pf1 = tau_df_pf_.loc[::2, wt1]
    #pf2 = tau_df_pf_.loc[::2, wt2]
    yrs_lab = np.arange(0,71,10)
    yrs_lab[0] = 1
    pf1 = tau_df_pf_.loc[yrs_lab,wt1]
    pf2 = tau_df_pf_.loc[yrs_lab,wt2]
    # Labeling of exponential line
    dates = [1, 10, 100, 500, 1000, 9000]
    exp1 = tau_df_exp.loc[dates, wt1]
    exp2 = tau_df_exp.loc[dates, wt2]
    #--
    # Piston-Flow
    ax.plot(tau_df_pf[wt1], tau_df_pf[wt2], color='grey', zorder=5)
    ax.scatter(pf1, pf2, color='grey', marker='o', facecolors='none', s=8, zorder=5)
    [ax.text(pf1.iloc[i], pf2.iloc[i], pf2.index[i], color='grey', fontsize=12) for i in range(len(pf1))]
    # Exponential
    ax.plot(tau_df_exp[wt1], tau_df_exp[wt2], color='black', linestyle='--', zorder=5)
    ax.scatter(exp1, exp2, color='black', marker='D', facecolors='none', s=8, zorder=5)
    [ax.text(exp1.iloc[i], exp2.iloc[i], exp2.index[i], fontsize=12) for i in range(1,len(exp1))]
    # Field Data
    for i in range(len(wells)):
        w  = wells[i]
        if j == 0:
            #ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
            ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
        elif j == 1:
            xx = ens_dict[wt1][w].to_numpy().ravel()[:20000] 
            yy = ens_dict[wt2][w].to_numpy().ravel()[:20000]
            #sns.kdeplot(x=xx,y=yy, ax=ax, cmap=cmlist[i], thresh=0.05, zorder=3, shade=True)
            ax.scatter(xx, yy, color='C{}'.format(i), marker='.', alpha=0.5, zorder=3)
        # Clean up
        #ax.set_yscale('log')
        #ax.set_xscale('log') 
    if j==0: 
        ax.set_ylabel('{} ({})'.format(ylabs[wt2], ylabs_units[wt2]))
    ax.set_xlabel('{} ({})'.format(ylabs[wt1], ylabs_units[wt1]))
    ax.set_xlim(-10, 126)
    ax.set_ylim(-90, 605)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    # Zoom in on right for inset
    if j==1:
        ax.set_xlim(-0.5, 11)
        ax.set_ylim(-5.0, 60)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.text(0.95, 0.95, 'Inset', horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='white', linewidth=0.1), zorder=10)
fig.legend(ncol=3,loc='center left', bbox_to_anchor=(0.25, 0.95), frameon=False)
plt.savefig('./figures/CFC113_CFC12_tracer_comp.svg', format='svg')
plt.show()


#--------------------------
# SF6 vs CFC12 
#--------------------------
fig, axes = plt.subplots(1,2,figsize=(7,3.5))
fig.subplots_adjust(top=0.88, bottom=0.17, left=0.15, right=0.96, hspace=0.35, wspace=0.2)
wells = ['PLM1', 'PLM7', 'PLM6']
wt1 = 'SF6'
wt2 = 'CFC12'
for j in range(2):
    ax = axes[j]
    # Labeling of pistonflow line
    tau_df_pf_ = tau_df_pf[tau_df_pf.index <= 76]
    #pf1 = tau_df_pf_.loc[::2, wt1]
    #pf2 = tau_df_pf_.loc[::2, wt2]
    yrs_lab = np.arange(0,71,10)
    yrs_lab[0] = 1
    pf1 = tau_df_pf_.loc[yrs_lab,wt1]
    pf2 = tau_df_pf_.loc[yrs_lab,wt2]
    # Labeling of exponential line
    dates = [1, 10, 100, 500, 1000, 9000]
    exp1 = tau_df_exp.loc[dates, wt1]
    exp2 = tau_df_exp.loc[dates, wt2]
    #--
    # Piston-Flow
    ax.plot(tau_df_pf[wt1], tau_df_pf[wt2], color='grey', zorder=5)
    ax.scatter(pf1, pf2, color='grey', marker='o', facecolors='none', s=8, zorder=5)
    [ax.text(pf1.iloc[i], pf2.iloc[i], pf2.index[i], color='grey', fontsize=12) for i in range(len(pf1))]
    # Exponential
    ax.plot(tau_df_exp[wt1], tau_df_exp[wt2], color='black', linestyle='--', zorder=5)
    ax.scatter(exp1, exp2, color='black', marker='D', facecolors='none', s=8, zorder=5)
    [ax.text(exp1.iloc[i], exp2.iloc[i], exp2.index[i], fontsize=12) for i in range(1,len(exp1))]
    # Field Data
    for i in range(len(wells)):
        w  = wells[i]
        if j == 0:
            #ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
            ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
        elif j == 1:
            xx = ens_dict[wt1][w].to_numpy().ravel()[:20000]
            yy = ens_dict[wt2][w].to_numpy().ravel()[:20000]
            sns.kdeplot(x=xx,y=yy, ax=ax, cmap=cmlist[i], thresh=0.05, zorder=3, shade=True)
            #ax.scatter(ens_dict[wt1][w][:2500], ens_dict[wt2][w][:2500], color='C{}'.format(i), marker='.', alpha=0.5, zorder=3)
        # Clean up
        #ax.set_yscale('log')
        #ax.set_xscale('log') 
    if j==0: 
        ax.set_ylabel('{} ({})'.format(ylabs[wt2], ylabs_units[wt2]))
    ax.set_xlabel('{} ({})'.format(ylabs[wt1], ylabs_units[wt1]))
    ax.set_xlim(-2.5, 21)
    ax.set_ylim(-90, 605)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    # Zoom in on right for inset
    if j==1:
        ax.set_xlim(-0.5, 2.05)
        ax.set_ylim(-5.0, 55)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.text(0.95, 0.95, 'Inset', horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='white', linewidth=0.1), zorder=10)
fig.legend(ncol=3,loc='center left', bbox_to_anchor=(0.25, 0.95), frameon=False)
plt.savefig('./figures/SF6_CFC12_tracer_comp.svg', format='svg')
plt.show()

    
#--------------------------
# He4 vs CFC12 
#--------------------------
fig, axes = plt.subplots(1,2,figsize=(7,3.5))
fig.subplots_adjust(top=0.88, bottom=0.17, left=0.15, right=0.96, hspace=0.35, wspace=0.2)
wells = ['PLM1', 'PLM7', 'PLM6']
wt1 = 'He4_ter'
wt2 = 'CFC12'
for j in range(2):
    ax = axes[j]
    ax.set_xscale('log')
    # Labeling of pistonflow line
    #tau_df_pf_ = tau_df_pf[tau_df_pf.index <= 76]
    #pf1 = tau_df_pf_.loc[::2, wt1]
    #pf2 = tau_df_pf_.loc[::2, wt2]
    dates = [1,10,20,30,40,50,60,70,100, 500, 1000, 9000]
    pf1 = tau_df_pf.loc[dates, wt1]
    pf2 = tau_df_pf.loc[dates, wt2]
    # Labeling of exponential line
    dates = [1, 10, 100, 500, 1000, 9000]
    exp1 = tau_df_exp.loc[dates, wt1]
    exp2 = tau_df_exp.loc[dates, wt2]
    #--
    # Piston-Flow
    ax.plot(tau_df_pf[wt1], tau_df_pf[wt2], color='grey', zorder=5)
    ax.scatter(pf1, pf2, color='grey', marker='o', facecolors='none', s=8, zorder=5)
    [ax.text(pf1.iloc[i], pf2.iloc[i], pf2.index[i], color='grey', fontsize=12) for i in range(len(pf1))]
    # Exponential
    ax.plot(tau_df_exp[wt1], tau_df_exp[wt2], color='black', linestyle='--', zorder=5)
    ax.scatter(exp1, exp2, color='black', marker='D', facecolors='none', s=8, zorder=5)
    [ax.text(exp1.iloc[i], exp2.iloc[i], exp2.index[i], fontsize=12) for i in range(1,len(exp1))]
    # Field Data
    for i in range(len(wells)):
        w  = wells[i]
        if j == 0:
            #ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
            ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
        elif j == 1:
            xx = ens_dict[wt1][w].to_numpy().ravel()[:20000]
            yy = ens_dict[wt2][w].to_numpy().ravel()[:20000]
            sns.kdeplot(x=xx,y=yy, ax=ax, log_scale=(True,False), cmap=cmlist[i], thresh=0.01, zorder=3, shade=True)
            #ax.scatter(ens_dict[wt1][w][:2500], ens_dict[wt2][w][:2500], color='C{}'.format(i), marker='.', alpha=0.5, zorder=3)
        # Clean up
        #ax.set_yscale('log')
        #ax.set_xscale('log') 
    if j==0: 
        ax.set_ylabel('{} ({})'.format(ylabs[wt2], ylabs_units[wt2]))
    ax.set_xlabel('{} ({})'.format(ylabs[wt1], ylabs_units[wt1]))
    ax.set_xlim(1.e-11, 1.e-6)
    ax.set_ylim(-50, 605)
    #ax.xaxis.set_major_locator(MultipleLocator(2))
    #ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    # Zoom in on right for inset
    if j==1:
        ax.set_xlim(1.e-9, 1.e-6)
        ax.set_ylim(-5.0, 55)
        #ax.xaxis.set_major_locator(MultipleLocator(1))
        #ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.text(0.95, 0.95, 'Inset', horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='white', linewidth=0.1))
fig.legend(ncol=3,loc='center left', bbox_to_anchor=(0.25, 0.95), frameon=False)
plt.savefig(os.path.join(fdir,'He4_CFC12_tracer_comp.svg'), format='svg')
plt.show()

    
#--------------------------
# He4 vs CFC11
#--------------------------
fig, axes = plt.subplots(1,2,figsize=(7,3.5))
fig.subplots_adjust(top=0.88, bottom=0.17, left=0.15, right=0.96, hspace=0.35, wspace=0.2)
wells = ['PLM1', 'PLM7', 'PLM6']
wt1 = 'He4_ter'
wt2 = 'CFC11'
for j in range(2):
    ax = axes[j]
    ax.set_xscale('log')
    # Labeling of pistonflow line
    #tau_df_pf_ = tau_df_pf[tau_df_pf.index <= 76]
    #pf1 = tau_df_pf_.loc[::2, wt1]
    #pf2 = tau_df_pf_.loc[::2, wt2]
    dates = [1,10,20,30,40,50,60,70,100, 500, 1000, 9000]
    pf1 = tau_df_pf.loc[dates, wt1]
    pf2 = tau_df_pf.loc[dates, wt2]
    # Labeling of exponential line
    dates = [1, 10, 100, 500, 1000, 9000]
    exp1 = tau_df_exp.loc[dates, wt1]
    exp2 = tau_df_exp.loc[dates, wt2]
    #--
    # Piston-Flow
    ax.plot(tau_df_pf[wt1], tau_df_pf[wt2], color='grey', zorder=5)
    ax.scatter(pf1, pf2, color='grey', marker='o', facecolors='none', s=8, zorder=5)
    [ax.text(pf1.iloc[i], pf2.iloc[i], pf2.index[i], color='grey', fontsize=12) for i in range(len(pf1))]
    # Exponential
    ax.plot(tau_df_exp[wt1], tau_df_exp[wt2], color='black', linestyle='--', zorder=5)
    ax.scatter(exp1, exp2, color='black', marker='D', facecolors='none', s=8, zorder=5)
    [ax.text(exp1.iloc[i], exp2.iloc[i], exp2.index[i], fontsize=12) for i in range(1,len(exp1))]
    # Field Data
    for i in range(len(wells)):
        w  = wells[i]
        if j == 0:
            #ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
            ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
        elif j == 1:
            xx = ens_dict[wt1][w].to_numpy().ravel()[:20000]
            yy = ens_dict[wt2][w].to_numpy().ravel()[:20000]
            sns.kdeplot(x=xx,y=yy, ax=ax, log_scale=(True,False), cmap=cmlist[i], thresh=0.01, zorder=3, shade=True)
            #ax.scatter(ens_dict[wt1][w][:2500], ens_dict[wt2][w][:2500], color='C{}'.format(i), marker='.', alpha=0.5, zorder=3)
        # Clean up
        #ax.set_yscale('log')
        #ax.set_xscale('log') 
    if j==0: 
        ax.set_ylabel('{} ({})'.format(ylabs[wt2], ylabs_units[wt2]))
    ax.set_xlabel('{} ({})'.format(ylabs[wt1], ylabs_units[wt1]))
    ax.set_xlim(1.e-11, 1.e-6)
    ax.set_ylim(-20, 305)
    #ax.xaxis.set_major_locator(MultipleLocator(2))
    #ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(25))
    # Zoom in on right for inset
    if j==1:
        ax.set_xlim(1.e-9, 1.e-6)
        ax.set_ylim(-3.0, 31)
        #ax.xaxis.set_major_locator(MultipleLocator(1))
        #ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.text(0.95, 0.95, 'Inset', horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='white', linewidth=0.1))
fig.legend(ncol=3,loc='center left', bbox_to_anchor=(0.25, 0.95), frameon=False)
plt.savefig('./figures/He4_CFC11_tracer_comp.svg', format='svg')
plt.show()


#--------------------------
# He4 vs 3H/3He 
#--------------------------
fig, axes = plt.subplots(1,2,figsize=(7,3.5))
fig.subplots_adjust(top=0.88, bottom=0.17, left=0.15, right=0.96, hspace=0.35, wspace=0.2)
wells = ['PLM1', 'PLM7', 'PLM6']
wt1 = 'He4_ter'
wt2 = 'H3_He3'
for j in range(2):
    ax = axes[j]
    ax.set_xscale('log')
    # Labeling of pistonflow line
    #tau_df_pf_ = tau_df_pf[tau_df_pf.index <= 76]
    #pf1 = tau_df_pf_.loc[::2, wt1]
    #pf2 = tau_df_pf_.loc[::2, wt2]
    dates = [1,10,20,30,40,50,60,70,100, 500, 1000, 9000]
    pf1 = tau_df_pf.loc[dates, wt1]
    pf2 = tau_df_pf.loc[dates, wt2]
    # Labeling of exponential line
    dates = [1, 10, 100, 500, 1000, 9000]
    exp1 = tau_df_exp.loc[dates, wt1]
    exp2 = tau_df_exp.loc[dates, wt2]
    #--
    # Piston-Flow
    ax.plot(tau_df_pf[wt1], tau_df_pf[wt2], color='grey', zorder=5)
    ax.scatter(pf1, pf2, color='grey', marker='o', facecolors='none', s=8, zorder=5)
    [ax.text(pf1.iloc[i], pf2.iloc[i], pf2.index[i], color='grey', fontsize=12) for i in range(len(pf1))]
    # Exponential
    ax.plot(tau_df_exp[wt1], tau_df_exp[wt2], color='black', linestyle='--', zorder=5)
    ax.scatter(exp1, exp2, color='black', marker='D', facecolors='none', s=8, zorder=5)
    [ax.text(exp1.iloc[i], exp2.iloc[i], exp2.index[i], fontsize=12) for i in range(1,len(exp1))]
    # Field Data
    for i in range(len(wells)):
        w  = wells[i]
        if j == 0:
            #ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
            ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
        elif j == 1:
            xx = ens_dict[wt1][w].to_numpy().ravel()[:20000]
            yy = ens_dict[wt2][w].to_numpy().ravel()[:20000]
            sns.kdeplot(x=xx,y=yy, ax=ax, cmap=cmlist[i], thresh=0.01, zorder=3, shade=True)
            #ax.scatter(ens_dict[wt1][w][:2500], ens_dict[wt2][w][:2500], color='C{}'.format(i), marker='.', alpha=0.5, zorder=3)
        # Clean up
        #ax.set_yscale('log')
        #ax.set_xscale('log') 
    if j==0: 
        ax.set_ylabel('{} ({})'.format(ylabs[wt2], ylabs_units[wt2]))
    ax.set_xlabel('{} ({})'.format(ylabs[wt1], ylabs_units[wt1]))
    ax.set_xlim(1.e-11, 1.e-6)
    ax.set_ylim(-1, 23)
    #ax.xaxis.set_major_locator(MultipleLocator(2))
    #ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    # Zoom in on right for inset
    if j==1:
        ax.set_xlim(1.e-9, 1.e-6)
        ax.set_ylim(-0.05, 0.4)
        #ax.xaxis.set_major_locator(MultipleLocator(1))
        #ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.text(0.95, 0.95, 'Inset', horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='white', linewidth=0.1))
fig.legend(ncol=3,loc='center left', bbox_to_anchor=(0.25, 0.95), frameon=False)
plt.savefig('./figures/He4_He3_tracer_comp.svg', format='svg')
plt.show()



#--------------------------
# He4 vs 3H
#--------------------------
fig, axes = plt.subplots(1,2,figsize=(7,3.5))
fig.subplots_adjust(top=0.88, bottom=0.17, left=0.15, right=0.96, hspace=0.35, wspace=0.2)
wells = ['PLM1', 'PLM7', 'PLM6']
wt1 = 'He4_ter'
wt2 = 'H3'
for j in range(2):
    ax = axes[j]
    ax.set_xscale('log')
    # Labeling of pistonflow line
    #tau_df_pf_ = tau_df_pf[tau_df_pf.index <= 76]
    #pf1 = tau_df_pf_.loc[::2, wt1]
    #pf2 = tau_df_pf_.loc[::2, wt2]
    dates = [1,10,20,30,40,50,60,70,100, 500, 1000, 9000]
    pf1 = tau_df_pf.loc[dates, wt1]
    pf2 = tau_df_pf.loc[dates, wt2]
    # Labeling of exponential line
    dates = [1, 10, 100, 500, 1000, 9000]
    exp1 = tau_df_exp.loc[dates, wt1]
    exp2 = tau_df_exp.loc[dates, wt2]
    #--
    # Piston-Flow
    ax.plot(tau_df_pf[wt1], tau_df_pf[wt2], color='grey', zorder=5)
    ax.scatter(pf1, pf2, color='grey', marker='o', facecolors='none', s=8, zorder=5)
    [ax.text(pf1.iloc[i], pf2.iloc[i], pf2.index[i], color='grey', fontsize=12) for i in range(len(pf1))]
    # Exponential
    ax.plot(tau_df_exp[wt1], tau_df_exp[wt2], color='black', linestyle='--', zorder=5)
    ax.scatter(exp1, exp2, color='black', marker='D', facecolors='none', s=8, zorder=5)
    [ax.text(exp1.iloc[i], exp2.iloc[i], exp2.index[i], fontsize=12) for i in range(1,len(exp1))]
    # Field Data
    for i in range(len(wells)):
        w  = wells[i]
        if j == 0:
            #ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
            ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
        elif j == 1:
            xx = ens_dict[wt1][w].to_numpy().ravel()[:20000]
            yy = ens_dict[wt2][w].to_numpy().ravel()[:20000]
            sns.kdeplot(x=xx,y=yy, ax=ax, cmap=cmlist[i], thresh=0.01, zorder=3, shade=True)
            #ax.scatter(ens_dict[wt1][w][:2500], ens_dict[wt2][w][:2500], color='C{}'.format(i), marker='.', alpha=0.5, zorder=3)
        # Clean up
        #ax.set_yscale('log')
        #ax.set_xscale('log') 
    if j==0: 
        ax.set_ylabel('{} ({})'.format(ylabs[wt2], ylabs_units[wt2]))
    ax.set_xlabel('{} ({})'.format(ylabs[wt1], ylabs_units[wt1]))
    ax.set_xlim(1.e-11, 1.e-6)
    ax.set_ylim(-5, 45)
    #ax.xaxis.set_major_locator(MultipleLocator(2))
    #ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    # Zoom in on right for inset
    if j==1:
        ax.set_xlim(1.e-9, 1.e-6)
        ax.set_ylim(-1, 12)
        #ax.xaxis.set_major_locator(MultipleLocator(1))
        #ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.text(0.95, 0.95, 'Inset', horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='white', linewidth=0.1))
fig.legend(ncol=3,loc='center left', bbox_to_anchor=(0.25, 0.95), frameon=False)
plt.savefig('./figures/He4_H3_tracer_comp.svg', format='svg')
plt.show()


#--------------------------
# CFC12 vs 3H
#--------------------------
fig, axes = plt.subplots(1,2,figsize=(7,3.5))
fig.subplots_adjust(top=0.88, bottom=0.17, left=0.15, right=0.96, hspace=0.35, wspace=0.2)
wells = ['PLM1', 'PLM7', 'PLM6']
wt1 = 'H3'
wt2 = 'CFC12'
for j in range(2):
    ax = axes[j]
    # Labeling of pistonflow line
    tau_df_pf_ = tau_df_pf[tau_df_pf.index <= 76]
    #pf1 = tau_df_pf_.loc[::2, wt1]
    #pf2 = tau_df_pf_.loc[::2, wt2]
    yrs_lab = np.arange(0,71,10)
    yrs_lab[0] = 1
    pf1 = tau_df_pf_.loc[yrs_lab,wt1]
    pf2 = tau_df_pf_.loc[yrs_lab,wt2]
    # Labeling of exponential line
    dates = [1, 10, 100, 500, 1000, 9000]
    exp1 = tau_df_exp.loc[dates, wt1]
    exp2 = tau_df_exp.loc[dates, wt2]
    #--
    # Piston-Flow
    ax.plot(tau_df_pf[wt1], tau_df_pf[wt2], color='grey', zorder=5)
    ax.scatter(pf1, pf2, color='grey', marker='o', facecolors='none', s=8, zorder=5)
    [ax.text(pf1.iloc[i], pf2.iloc[i], pf2.index[i], color='grey', fontsize=12) for i in range(len(pf1))]
    # Exponential
    ax.plot(tau_df_exp[wt1], tau_df_exp[wt2], color='black', linestyle='--', zorder=5)
    ax.scatter(exp1, exp2, color='black', marker='D', facecolors='none', s=8, zorder=5)
    [ax.text(exp1.iloc[i], exp2.iloc[i], exp2.index[i], fontsize=12) for i in range(1,len(exp1))]
    # Field Data
    for i in range(len(wells)):
        w  = wells[i]
        if j == 0:
            #ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
            ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
        elif j == 1:
            xx = ens_dict[wt1][w].to_numpy().ravel()[:20000]
            yy = ens_dict[wt2][w].to_numpy().ravel()[:20000]
            sns.kdeplot(x=xx,y=yy, ax=ax, cmap=cmlist[i], thresh=0.01, zorder=3, shade=True)
            #ax.scatter(ens_dict[wt1][w][:2500], ens_dict[wt2][w][:2500], color='C{}'.format(i), marker='.', alpha=0.5, zorder=3)
        # Clean up
        #ax.set_yscale('log')
        #ax.set_xscale('log') 
    if j==0: 
        ax.set_ylabel('{} ({})'.format(ylabs[wt2], ylabs_units[wt2]))
    ax.set_xlabel('{} ({})'.format(ylabs[wt1], ylabs_units[wt1]))
    ax.set_xlim(-5, 45)
    ax.set_ylim(-90, 605)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    # Zoom in on right for inset
    if j==1:
        ax.set_xlim(-1, 12)
        ax.set_ylim(-5.0, 55)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.text(0.95, 0.95, 'Inset', horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='white', linewidth=0.1), zorder=10)
fig.legend(ncol=3,loc='center left', bbox_to_anchor=(0.25, 0.95), frameon=False)
plt.savefig('./figures/H3_CFC12_tracer_comp.svg', format='svg')
plt.show()



#--------------------------
# CFC12 vs 3H/He3
#--------------------------
fig, axes = plt.subplots(1,2,figsize=(7,3.5))
fig.subplots_adjust(top=0.88, bottom=0.17, left=0.15, right=0.96, hspace=0.35, wspace=0.2)
wells = ['PLM1', 'PLM7', 'PLM6']
wt1 = 'H3_He3'
wt2 = 'CFC12'
for j in range(2):
    ax = axes[j]
    # Labeling of pistonflow line
    tau_df_pf_ = tau_df_pf[tau_df_pf.index <= 76]
    #pf1 = tau_df_pf_.loc[::2, wt1]
    #pf2 = tau_df_pf_.loc[::2, wt2]
    yrs_lab = np.arange(0,71,10)
    yrs_lab[0] = 1
    pf1 = tau_df_pf_.loc[yrs_lab,wt1]
    pf2 = tau_df_pf_.loc[yrs_lab,wt2]
    # Labeling of exponential line
    dates = [1, 10, 100, 500, 1000, 9000]
    exp1 = tau_df_exp.loc[dates, wt1]
    exp2 = tau_df_exp.loc[dates, wt2]
    #--
    # Piston-Flow
    ax.plot(tau_df_pf[wt1], tau_df_pf[wt2], color='grey', zorder=5)
    ax.scatter(pf1, pf2, color='grey', marker='o', facecolors='none', s=8, zorder=5)
    [ax.text(pf1.iloc[i], pf2.iloc[i], pf2.index[i], color='grey', fontsize=12) for i in range(len(pf1))]
    # Exponential
    ax.plot(tau_df_exp[wt1], tau_df_exp[wt2], color='black', linestyle='--', zorder=5)
    ax.scatter(exp1, exp2, color='black', marker='D', facecolors='none', s=8, zorder=5)
    [ax.text(exp1.iloc[i], exp2.iloc[i], exp2.index[i], fontsize=12) for i in range(1,len(exp1))]
    # Field Data
    for i in range(len(wells)):
        w  = wells[i]
        if j == 0:
            #ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
            ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
        elif j == 1:
            xx = ens_dict[wt1][w].to_numpy().ravel()[:20000]
            yy = ens_dict[wt2][w].to_numpy().ravel()[:20000]
            sns.kdeplot(x=xx,y=yy, ax=ax, cmap=cmlist[i], thresh=0.01, zorder=3, shade=True)
            #ax.scatter(ens_dict[wt1][w][:2500], ens_dict[wt2][w][:2500], color='C{}'.format(i), marker='.', alpha=0.5, zorder=3)
        # Clean up
        #ax.set_yscale('log')
        #ax.set_xscale('log') 
    if j==0: 
        ax.set_ylabel('{} ({})'.format(ylabs[wt2], ylabs_units[wt2]))
    ax.set_xlabel('{} ({})'.format(ylabs[wt1], ylabs_units[wt1]))
    ax.set_xlim(-5, 30)
    ax.set_ylim(-90, 605)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    # Zoom in on right for inset
    if j==1:
        ax.set_xlim(-0.05, 0.4)
        ax.set_ylim(-5.0, 55)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.025))
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.text(0.95, 0.95, 'Inset', horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='white', linewidth=0.1), zorder=10)
fig.legend(ncol=3,loc='center left', bbox_to_anchor=(0.25, 0.95), frameon=False)
plt.savefig('./figures/He3_CFC12_tracer_comp.svg', format='svg')
plt.show()





#--------------------------
# 3H vs 3H/He3
#--------------------------
fig, axes = plt.subplots(1,2,figsize=(7,3.5))
fig.subplots_adjust(top=0.88, bottom=0.17, left=0.15, right=0.96, hspace=0.35, wspace=0.2)
wells = ['PLM1', 'PLM7', 'PLM6']
wt1 = 'H3_He3'
wt2 = 'H3'
for j in range(2):
    ax = axes[j]
    # Labeling of pistonflow line
    tau_df_pf_ = tau_df_pf[tau_df_pf.index <= 76]
    #pf1 = tau_df_pf_.loc[::2, wt1]
    #pf2 = tau_df_pf_.loc[::2, wt2]
    yrs_lab = np.arange(0,71,10)
    yrs_lab[0] = 1
    pf1 = tau_df_pf_.loc[yrs_lab,wt1]
    pf2 = tau_df_pf_.loc[yrs_lab,wt2]
    # Labeling of exponential line
    dates = [1, 10, 100, 500, 1000, 9000]
    exp1 = tau_df_exp.loc[dates, wt1]
    exp2 = tau_df_exp.loc[dates, wt2]
    #--
    # Piston-Flow
    ax.plot(tau_df_pf[wt1], tau_df_pf[wt2], color='grey', zorder=5)
    ax.scatter(pf1, pf2, color='grey', marker='o', facecolors='none', s=8, zorder=5)
    [ax.text(pf1.iloc[i], pf2.iloc[i], pf2.index[i], color='grey', fontsize=12) for i in range(len(pf1))]
    # Exponential
    ax.plot(tau_df_exp[wt1], tau_df_exp[wt2], color='black', linestyle='--', zorder=5)
    ax.scatter(exp1, exp2, color='black', marker='D', facecolors='none', s=8, zorder=5)
    [ax.text(exp1.iloc[i], exp2.iloc[i], exp2.index[i], fontsize=12) for i in range(1,len(exp1))]
    # Field Data
    for i in range(len(wells)):
        w  = wells[i]
        if j == 0:
            #ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
            ax.scatter(ens_dict[wt1][w].mean(), ens_dict[wt2][w].mean(), color='C{}'.format(i), marker='o', alpha=1.0, zorder=7, label=w.split('_')[0])
        elif j == 1:
            xx = ens_dict[wt1][w].to_numpy().ravel()[:20000]
            yy = ens_dict[wt2][w].to_numpy().ravel()[:20000]
            sns.kdeplot(x=xx,y=yy, ax=ax, cmap=cmlist[i], thresh=0.01, zorder=3, shade=True)
            #ax.scatter(ens_dict[wt1][w][:2500], ens_dict[wt2][w][:2500], color='C{}'.format(i), marker='.', alpha=0.5, zorder=3)
        # Clean up
        #ax.set_yscale('log')
        #ax.set_xscale('log') 
    if j==0: 
        ax.set_ylabel('{} ({})'.format(ylabs[wt2], ylabs_units[wt2]))
    ax.set_xlabel('{} ({})'.format(ylabs[wt1], ylabs_units[wt1]))
    ax.set_xlim(-5, 30)
    ax.set_ylim(-5, 45)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    # Zoom in on right for inset
    if j==1:
        ax.set_xlim(-0.05, 0.4)
        ax.set_ylim(-0.1, 12)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.025))
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.text(0.95, 0.95, 'Inset', horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='white', linewidth=0.1), zorder=10)
fig.legend(ncol=3,loc='center left', bbox_to_anchor=(0.25, 0.95), frameon=False)
plt.savefig('./figures/He3_H3_tracer_comp.svg', format='svg')
plt.show()













