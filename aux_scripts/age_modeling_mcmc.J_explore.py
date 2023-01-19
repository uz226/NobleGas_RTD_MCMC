# This scripts explores the 4He production rate priors
# Need to run age_modeling_mcmc.prep.py first


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

sys.path.insert(0, '../utils')
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
map_dict  = pd.read_pickle('../map_dict.pk')
ens_dict  = pd.read_pickle('../ens_dict.pk')
C_in_dict = pd.read_pickle('../C_in_dict.pk')



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



#------------------------------------------------------
# 
# J_He Prior predictions
#
#------------------------------------------------------
import scipy.stats as sc

N = 10000

# Release factor as beta distribution
del_mu = 1.0
del_sd = 5.0
Del_rv = sc.halfnorm.rvs(loc=del_mu, scale=del_sd, size=N)


# Uranium Concentration
U_mu   = 3.0
U_per  = 10/100.
U_rv = sc.norm.rvs(loc=U_mu, scale=U_mu*U_per, size=N)


# Thorium Concentration
Th_mu   = 10.0
Th_per  = 10/100.
Th_rv = sc.norm.rvs(loc=Th_mu, scale=Th_mu*Th_per, size=N)


# Matrix Porosity as normal
phi_mu  = 5.0  # percent
phi_sd  = 1.0
phi_rv  = sc.norm.rvs(loc=phi_mu, scale=phi_sd, size=N)
phi_rv = phi_rv/100.



# Now Simulate He accumation rates
J_prior = ng_utils.J_flux(Del=1.0, rho_r=2700, rho_w=1000, U=3.0, Th=10.0, phi=0.05)

J_ens = []
for Del, U, Th, phi in zip(Del_rv, U_rv, Th_rv, phi_rv):
    J_ens.append(ng_utils.J_flux(Del=Del, rho_r=2700, rho_w=1000, U=U, Th=Th, phi=phi))
J_ens = np.array(J_ens)



#
# Combined plots for all parameters
fig, ax = plt.subplots(nrows=5,ncols=1, figsize=(4,8))
# Release Factor
ax[0].hist(Del_rv, bins=30)
ax[0].set_xlabel(r'$\Lambda$ (-)')
ax[0].set_xlim(0, Del_rv.max()+1)
# Uranium
ax[1].hist(U_rv, bins=30)
ax[1].set_xlabel('[U] (ppm)')
# Thorium
ax[2].hist(Th_rv, bins=30)
ax[2].set_xlabel('[Th] (ppm)')
# Matrix porosity
ax[3].hist(phi_rv, bins=30)
ax[3].set_xlabel('Matrix Porosity (-)')
#
ax[4].hist(J_ens, bins=np.logspace(np.log10(J_ens.min()),np.log10(J_ens.max()), 30))
ax[4].set_xscale('log')
ax[4].set_xlabel(r'$J_{He} \ (cm^{3}STP\,g_{H2O}^{-1}\,year^{-1})$')
ax[4].tick_params(axis='y',which='both',left=False,labelleft=False)
# Cleanup
for i in range(5):
    ax[i].minorticks_on()
    ax[i].xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax[i].tick_params(axis='y',which='both',left=False,labelleft=False)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['left'].set_visible(False)
fig.tight_layout()
plt.savefig('./figures/He_production_params.png', dpi=300)
plt.show()



print (np.log10(J_ens).mean())
print (np.log10(J_ens).std())






