# Performs Monte Carlo Analysis on multiple different RTD models
# 
# Runs that age_modeling_mcmc.prep.py first


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
##from theano import as_op
#from theano.compile.ops import as_op
#import arviz as az


import copy
import seaborn as sns

from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pd.set_option("display.precision", 8)







# Plotting label utils
ylabs = {'CFC11':'CFC-11',
         'CFC12':'CFC-12',
         'CFC113':'CFC-113',
         'SF6':'SF$_{6}$',
         'He4_ter':'$^{4}$He$\mathrm{_{ter}}$',
         'H3_He3':'$\mathrm{^{3}H/^{3}He}$',
         'H3':'$\mathrm{^{3}H}$'}

ylabs_units = {'CFC11':'pptv',
               'CFC12':'pptv',
               'CFC113':'pptv',
               'SF6':'pptv',
               'He4_ter':r'cm$^{3}$STP/g',
               'H3_He3':'TU',
               'H3':'TU'}




# Plotting functions for error ellipsoids
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)



# Read in field data dictionaries
# Produced in age_modleing_mcmc.prep.py
map_dict   = pd.read_pickle('../map_dict.pk')
ens_dict   = pd.read_pickle('../ens_dict.pk')
C_in_dict  = pd.read_pickle('../C_in_dict.pk')





#------------------------------
#
# Define RTD Mixing Models
#
#------------------------------

n_samples  = 15000 # Total number of samples for Monte Carlo
tau_max    = 15000

#
# Model 1:  piston flow for single model
f1        = np.ones(n_samples)  # Fraction of component 1
f2        = 1-f1                # Needs to be zero for single component model
#tau_list1 = np.random.uniform(1, 10000, n_samples) # tau of component 1
tau_list1 = np.linspace(1, tau_max, n_samples) # tau of component 1
tau_list2 = np.random.uniform(-9999, -9999, n_samples) # Does not matter here
mod1_in   = pd.DataFrame(np.column_stack((f1,f2,tau_list1,tau_list2)), columns=['f1','f2','tau1','tau2'])


#
# Model 2:  exponential for single model
f1        = np.ones(n_samples)  # Fraction of component 1
f2        = 1-f1                # Needs to be zero for single component model
#tau_list1 = np.random.uniform(1, tau_max, n_samples) # tau of component 1
tau_list1 = np.linspace(1, tau_max, n_samples) # tau of component 1
tau_list2 = np.random.uniform(-9999, -9999, n_samples) # Does not matter here
mod2_in   = pd.DataFrame(np.column_stack((f1,f2,tau_list1,tau_list2)), columns=['f1','f2','tau1','tau2'])


#
# Model 3: exponential-pistonflow (EPM)
f1        = np.ones(n_samples)  # Fraction of component 1
f2        = 1-f1                # Needs to be zero for single component model
eta1      = np.random.uniform(1, 7, n_samples)
tau_list1 = np.random.uniform(1, tau_max, n_samples)   # tau of component 1
tau_list2 = np.random.uniform(-9999, -9999, n_samples) # Does not matter here
mod3_in   = pd.DataFrame(np.column_stack((f1,f2,tau_list1,tau_list2,eta1)), columns=['f1','f2','tau1','tau2','eta1'])


#
# Model 4: Dispersion Model
f1        = np.ones(n_samples)  # Fraction of component 1
f2        = 1-f1                # Needs to be zero for single component model
D1        = np.random.uniform(0.01, 2.0, n_samples)     # Dispersion Model factor -- inverse of Pe 
tau_list1 = np.random.uniform(1, tau_max, n_samples) # advective mean residence time
tau_list2 = np.random.uniform(-9999, -9999, n_samples) # Does not matter here
mod4_in   = pd.DataFrame(np.column_stack((f1,f2,tau_list1,tau_list2,D1)), columns=['f1','f2','tau1','tau2','D1'])

#
# Model 5: BMM1 pfm-pfm
f1        = np.random.uniform(0.1, 0.99, n_samples) # Fraction of component 1
f2        = 1-f1
tau_list1 = np.random.uniform(1, tau_max, n_samples) # tau of component 1
tau_list2 = np.random.uniform(1, tau_max, n_samples) # tau of componenet 2
mod5_in   = pd.DataFrame(np.column_stack((f1,f2,tau_list1,tau_list2)), columns=['f1','f2','tau1','tau2'])

#
# Model 6: BMM2 piston-exponential
f1        = np.random.uniform(0.1, 0.99, n_samples) # Fraction of component 1
f2        = 1-f1
tau_list1 = np.random.uniform(1, tau_max, n_samples) # tau of component 1
tau_list2 = np.random.uniform(1, tau_max, n_samples) # tau of componenet 2
mod6_in   = pd.DataFrame(np.column_stack((f1,f2,tau_list1,tau_list2)), columns=['f1','f2','tau1','tau2'])

#
# Model 7: BMM3 piston-exp_piston_flow 
# Assumes second component is exp_pist_flow
f1        = np.random.uniform(0.1, 0.99, n_samples) # Fraction of component 1
f2        = 1-f1
tau_list1 = np.random.uniform(1, tau_max, n_samples) # tau of component 1
tau_list2 = np.random.uniform(1, tau_max, n_samples) # tau of componenet 2
eta2      = np.random.uniform(1, 5, n_samples)
eta1      = np.zeros_like(eta2).astype(bool)
mod7_in   = pd.DataFrame(np.column_stack((f1,f2,tau_list1,tau_list2,eta1,eta2)), columns=['f1','f2','tau1','tau2','eta1','eta2'])

#
# Model 8: BMM3 piston-dipsersioin 
# Assumes second component is dispersion
f1        = np.random.uniform(0.1, 0.99, n_samples) # Fraction of component 1
f2        = 1-f1
tau_list1 = np.random.uniform(1, tau_max, n_samples) # tau of component 1
tau_list2 = np.random.uniform(1, tau_max, n_samples) # tau of componenet 2
D2        = np.random.uniform(0.01, 2.0, n_samples)     # Dispersion Model factor -- inverse of Pe 
D1        = np.zeros_like(D2).astype(bool)
mod8_in   = pd.DataFrame(np.column_stack((f1,f2,tau_list1,tau_list2,D1,D2)), columns=['f1','f2','tau1','tau2','D1','D2'])


# Making use that EPM model is = to emm at eta=0 and close to pfm at eta>5
#
# Model 9: epm-epm
f1        = np.random.uniform(0.1, 0.99, n_samples) # Fraction of component 1
f2        = 1-f1
tau_list1 = np.random.uniform(1, tau_max, n_samples) # tau of component 1
tau_list2 = np.random.uniform(1, tau_max, n_samples) # tau of componenet 2
eta1      = np.random.uniform(1, 7, n_samples)
eta2      = np.random.uniform(1, 7, n_samples)
mod9_in   = pd.DataFrame(np.column_stack((f1,f2,tau_list1,tau_list2,eta1,eta2)), columns=['f1','f2','tau1','tau2','eta1','eta2'])

# Making use that EPM model is = to emm at eta=0 and close to pfm at eta>5
#
# Model 10: epm-dm
f1        = np.random.uniform(0.1, 0.99, n_samples) # Fraction of component 1
f2        = 1-f1
tau_list1 = np.random.uniform(1, tau_max, n_samples) # tau of component 1
tau_list2 = np.random.uniform(1, tau_max, n_samples) # tau of componenet 2
eta1      = np.random.uniform(1, 7, n_samples)
eta2      = np.zeros_like(eta1).astype(bool)
D2        = np.random.uniform(0.01, 2.0, n_samples)     # Dispersion Model factor -- inverse of Pe 
D1        = np.zeros_like(D2).astype(bool)
mod10_in   = pd.DataFrame(np.column_stack((f1,f2,tau_list1,tau_list2,eta1,eta2,D1,D2)), columns=['f1','f2','tau1','tau2','eta1','eta2','D1','D2'])



#
# Model 11: Fractrue/Matrix Diffusion Model (FDM) -- from Gardner 2015
#f1        = np.ones(n_samples)  # Fraction of component 1
#f2        = 1-f1                # Needs to be zero for single component model
#D         = np.random.uniform(0.01,2.0, n_samples)     # Dispersion Model factor -- inverse of Pe 
#bbar      = 10**np.random.uniform(-5, -2, n_samples)   # Fracture aperture (m)
#Phi_im    = np.random.uniform(0.005, 0.05, n_samples)  # Immobile matrix porosity
#tau_list1 = np.random.uniform(1, 10000, n_samples) # advective mean residence time
#tau_list2 = np.random.uniform(-9999, -9999, n_samples) # Does not matter here
#mod11_in   = pd.DataFrame(np.column_stack((f1,f2,tau_list1,tau_list2,D,bbar,Phi_im)), columns=['f1','f2','tau1','tau2','D','bbar','Phi_im'])



# RTD naming to make things easier
mod_map = {'piston':'pfm',
           'exponential':'emm',
           'exp_pist_flow':'epm',
           'dispersion':'dm'}




#tlist = ['CFC11', 'CFC12', 'CFC113', 'SF6', 'He4_ter', 'H3', 'He3']
tlist = ['He4_ter', 'CFC12', 'H3','He3', 'SF6']

def run_bmm_model(bmm_df, mod_type1, mod_type2, J):
    '''Assumes bmm_df has columns:
        f1,f2,tau1,tau2'''
    #pdb.set_trace()
    bmm_df_out = bmm_df.copy() # Holds the outputs
    
    # Weighted mean age by fraction
    bmm_df_out['tau'] = bmm_df_out['tau1']*bmm_df_out['f1'] + bmm_df_out['tau2']*bmm_df_out['f2']
    
    # pad parameter dataframe to avoid KeyErrors for specific parameters
    pars = ['eta1','eta2','D1','D2','bbar','Phi_im']
    for p in pars:
        if p not in bmm_df_out.columns:
            bmm_df_out[p] = False
    
    # RTD model name
    if mod_type2:
        mod_type  = '{}-{}'.format(mod_map[mod_type1], mod_map[mod_type2])
    else:
        mod_type = mod_map[mod_type1]
        
    print ('--- {} ---'.format(mod_type))
    
    # Forward convolution 
    for t in tlist:
        print ('  {}...'.format(t))
        # Initialize Models
        if t in ['H3','He3']:
            tr_in = C_in_dict['H3'].copy()
        else:
            tr_in = C_in_dict[t].copy()
        conv_ = conv.tracer_conv_integral(C_t=tr_in, t_samp=tr_in.index[-1]) # first  comp of bmm
        # Run through parameters
        for i in range(len(bmm_df)):
            #pdb.set_trace()
            tau1 = bmm_df_out.loc[i,'tau1']
            tau2 = bmm_df_out.loc[i,'tau2']
            
            eta1   = bmm_df_out.loc[i,'eta1'] 
            eta2   = bmm_df_out.loc[i,'eta2'] 
            D1     = bmm_df_out.loc[i,'D1']
            D2     = bmm_df_out.loc[i,'D2']
            bbar   = bmm_df_out.loc[i,'bbar']
            Phi_im = bmm_df_out.loc[i,'Phi_im']

            #pdb.set_trace()            

            # First, set up convolution parameters
            if t == 'H3':
                conv_.update_pars(tau=tau1, mod_type=mod_type1, t_half=12.34, eta=eta1, D=D1, bbar=bbar, Phi_im=Phi_im)
            elif t == 'He3':
                conv_.update_pars(tau=tau1, mod_type=mod_type1, t_half=12.34, rad_accum='3He', eta=eta1, D=D1, bbar=bbar, Phi_im=Phi_im)
            elif t == 'He4_ter':
                if J:
                    pass
                else:
                    J = ng_utils.J_flux(Del=1., rho_r=2700, rho_w=1000, U=3.0, Th=10.0, phi=0.05) # sample of these parameters?
                conv_.update_pars(tau=tau1, mod_type=mod_type1, rad_accum='4He', J=J, eta=eta1, D=D1, bbar=bbar, Phi_im=Phi_im)
            else:
                conv_.update_pars(tau=tau1, mod_type=mod_type1, eta=eta1, D=D1, bbar=bbar, Phi_im=Phi_im)
            # Convolution to find model 1 concentration
            C1_ = conv_.convolve()
            bmm_df_out.loc[i, '{}_1'.format(t)] = C1_
            # Matrix Diffusion Model also has total transit time output
            #bmm_df_out.loc[i,'tau_trans'] = conv_.FM_mu
            # Update for model 2, skip if using single model
            if mod_type2:
                conv_.mod_type = mod_type2
                conv_.tau      = tau2
                conv_.eta      = eta2
                conv_.D        = D2
                C2_ = conv_.convolve()
                bmm_df_out.loc[i, '{}_2'.format(t)] = C2_
            else:
                C2_ = -99999.0 # dummy value that should not matter, as long as f2 is set to zero
            # Use fraction factors for weighted sum of concentrations
            C_ = C1_*bmm_df.loc[i,'f1'] + C2_*bmm_df.loc[i,'f2']
            bmm_df_out.loc[i,t] = C_
    bmm_df_out['H3_He3'] = bmm_df_out['H3']/bmm_df_out['He3']      
    # Include modtype flag in dataframe
    bmm_df_out.insert(loc=0, column='mod', value=len(bmm_df_out)*[mod_type])
    
    return bmm_df_out.copy()



run_rtds = True
if run_rtds:
    
    mod1_out = run_bmm_model(bmm_df=mod1_in.copy(), mod_type1='piston', mod_type2=False, J=False)
    mod2_out = run_bmm_model(mod2_in.copy(), 'exponential',   False, False)
    mod3_out = run_bmm_model(mod3_in.copy(), 'exp_pist_flow', False, False)
    mod4_out = run_bmm_model(mod4_in.copy(), 'dispersion',    False, False)
    
    #mod5_out = run_bmm_model(mod5_in.copy(), 'piston', 'piston', False)
    #mod6_out = run_bmm_model(mod6_in.copy(), 'piston', 'exponential', False)
    #mod7_out = run_bmm_model(mod7_in.copy(), 'piston', 'exp_pist_flow', False)
    #mod8_out = run_bmm_model(mod8_in.copy(), 'piston', 'dispersion', False)
    
    mod9_out  = run_bmm_model(mod9_in.copy(), 'exp_pist_flow', 'exp_pist_flow', False)
    mod10_out = run_bmm_model(mod10_in.copy(),'exp_pist_flow', 'dispersion', False)
    
    #mod11_out = run_bmm_model(mod7_in.copy(), 'frac_inf_diff', False)
    
    #------------------------------
    # Compile the models
    #------------------------------
    mod_df_master = pd.concat((mod1_out, 
                               mod2_out, 
                               mod3_out, 
                               mod4_out,
                               #mod5_out,
                               #mod6_out,
                               #mod7_out,
                               #mod8_out,
                               mod9_out,
                               mod10_out))
    mod_df_master.reset_index(drop=True, inplace=True)
    mod_df_master.index.name='run'
    mod_df_master.to_csv('rtd_df_master.csv', index=True)
    print ('---')
    
else:
    mod_df_master = pd.read_csv('./rtd_df_master.csv', index_col='run')
    print ('Found dataframe...')



#
# Clean up df
#
# Scale helium
mod_df_master.loc[:,'He4_ter'] *= 1.e8

# Set detection limit to avoid 10^-100 TU
cfc12_bdl = np.sort(np.unique(C_in_dict['CFC12']))[1]*0.9
mod_df_master.loc[:,'CFC12'][mod_df_master.loc[:,'CFC12']<cfc12_bdl] = cfc12_bdl
#
trit_bdl = 1.e-2
mod_df_master.loc[:,'H3'][mod_df_master.loc[:,'H3']<trit_bdl] = trit_bdl




# Just to clean up labels
name_map = {'pfm':'PFM',
            'emm':'EMM',
            'epm':'EPM',
            'dm':'DM'}




#-------------------------------------------
#
# Plotting
#
#-------------------------------------------


def pull_lab_xy(rt, dff, tr):
    ind = abs(dff['tau1']-rt).idxmin()
    return dff.loc[ind,tr]    



#
#
# Young tracers
#
#

tlist = ['H3','SF6','CFC12']#, 'H3_He3'] 
wells = ['PLM1', 'PLM7', 'PLM6'] 
cmb = list(combinations(tlist, 2))

ms = ['o','D', '<']
ms = ['o','D', 'P']
fill = ['y','white', 'black']

# Pick RTD models to plot
# reconfigure the list
#mod_list    = [1,2,3,4,9,10] # Order to plot in
#mod_names   = ['pfm','emm','epm','dm','epm-epm','epm-dm'] 
mod_list    = [1,2,3,9] # Order to plot in
mod_names   = ['pfm','emm','epm','epm-epm'] 


mask = [i in mod_names for i in mod_df_master['mod'].to_list()]

mod_df_ = mod_df_master.copy()[mask]
mod_df_.reset_index(drop=True, inplace=True)


fig, axes = plt.subplots(nrows=3, ncols=len(mod_names), figsize=(8, 6))
fig.subplots_adjust(top=0.92, bottom=0.1, left=0.08, right=0.88, hspace=0.5, wspace=0.12)
for k in range(len(cmb)):

    # Plot tracer-tracer scatter plots
    t1 = cmb[k][0]
    t2 = cmb[k][1]
    
    #ax.scatter(mod_df_master[t1], mod_df_master[t2], marker='.', color='black')
    # Plot individual models seperate
    for m in range(len(mod_names)):        
        ax = axes[k,m]
        
        mm = mod_names[m]
        dd = copy.deepcopy(mod_df_)[mod_df_['mod'] == mm]
        
        # Plot pfm and emm as lines - not scatter points
        if mm in ['pfm','emm']:
            # plot just for single rtd
            ax.plot(dd[t1], dd[t2], marker='None', color='C{}'.format(m), linewidth=2.0, alpha=1.0, zorder=5)
            rt_pfm = [1, 10, 20, 30, 40, 50, 60, 70]
            rt_emm = [1, 10, 100, 250, 1000, 10000]
            if mm == 'pfm':
                #[ax.text(pull_lab_xy(i,dd,t1), pull_lab_xy(i,dd,t2), i, color='black', fontsize=10, 
                #         horizontalalignment='center', verticalalignment='center', zorder=5) for i in rt_pfm]
                [ax.annotate(i, xy=(pull_lab_xy(i,dd,t1),pull_lab_xy(i,dd,t2)), xycoords='data', color='black', fontsize=10, 
                         horizontalalignment='center', verticalalignment='center', zorder=5) for i in rt_pfm]
            if mm == 'emm':
                #[ax.text(pull_lab_xy(i,dd,t1), pull_lab_xy(i,dd,t2), i, color='black', fontsize=10,
                #         horizontalalignment='center', verticalalignment='center', zorder=5) for i in rt_emm]
                [ax.annotate(i, xy=(pull_lab_xy(i,dd,t1),pull_lab_xy(i,dd,t2)), xycoords='data', color='black', fontsize=10, 
                         horizontalalignment='center', verticalalignment='center', zorder=5) for i in rt_emm]
        else:
            ax.scatter(dd[t1], dd[t2], marker='.', color='C{}'.format(m), alpha=0.6, zorder=5)    
        
        # plot all points
        ax.scatter(mod_df_[t1], mod_df_[t2], marker='.', color='grey', alpha=0.1)
    
        if k == 0:
            try:
                mm_ = mm.split('-')
                ax.set_title('{}-{}'.format(name_map[mm_[0]], name_map[mm_[1]]))
            except IndexError:
                ax.set_title(name_map[mm])
    
        #
        # Observations
        for w in range(len(wells)):
            xx = copy.deepcopy(ens_dict)[t1][wells[w]].to_numpy().ravel()#[:20000]
            yy = copy.deepcopy(ens_dict)[t2][wells[w]].to_numpy().ravel()#[:20000]
            #sns.kdeplot(x=xx,y=yy, ax=ax, cmap=cmlist[w], thresh=0.05, zorder=3, shade=True)
            if t1 == 'He4_ter':
                xx *= 1.e8
                ax.scatter(xx.mean(), yy.mean(), color='black', marker=ms[w], s=50.0, alpha=1.0, zorder=8, 
                           facecolors=fill[w], edgecolors='black', linewidth=1.5, label=wells[w].split('_')[0]) # no log use this
                #ax.scatter(xx.mean(), yy.mean(), color='black', marker=ms[w], alpha=1.0, zorder=6, label=wells[w].split('_')[0]) # no log use this
                #confidence_ellipse(xx, yy, ax, edgecolor='black', alpha=0.8, zorder=10)
                ax.errorbar(xx.mean(), yy.mean(), xerr=xx.std()*3, yerr=yy.std()*3, alpha=0.6, ecolor='black',zorder=6)
            elif t2 == 'He4_ter':
                yy *= 1.e8
                #ax.scatter(xx.mean(), yy.mean(), color='black', marker=ms[w], alpha=1.0, zorder=6, label=wells[w].split('_')[0])
                ax.scatter(xx.mean(), yy.mean(), color='black', marker=ms[w], s=50.0, alpha=1.0, zorder=8, 
                           facecolors=fill[w], edgecolors='black', linewidth=1.5, label=wells[w].split('_')[0]) # no log use this
                #confidence_ellipse(xx, yy, ax, edgecolor='black', alpha=0.8, zorder=10)
                ax.errorbar(xx.mean(), yy.mean(), xerr=xx.std()*3, yerr=yy.std()*3, alpha=0.6, ecolor='black')
            else:
                #ax.scatter(xx.mean(), yy.mean(), color='black', marker=ms[w], facecolors='black', edgecolors='black',
                #           alpha=1.0, zorder=6, label=wells[w].split('_')[0])
                ax.scatter(xx.mean(), yy.mean(), color='black', marker=ms[w], s=50.0, alpha=1.0, zorder=8, 
                           facecolors=fill[w], edgecolors='black', linewidth=1.5, label=wells[w].split('_')[0]) # no log use this
                #confidence_ellipse(xx, yy, ax, edgecolor='black', alpha=0.8, zorder=10)
                ax.errorbar(xx.mean(), yy.mean(), xerr=xx.std()*3, yerr=yy.std()*3, alpha=0.6, ecolor='black',zorder=6)
        # Clean-up
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        
        if t1 == 'CFC12':
            ax.set_xlim(-2,60)
            ax.xaxis.set_major_locator(MultipleLocator(25))
            pass
        if t2 == 'CFC12':
            ax.set_ylim(-2,60)
            ax.yaxis.set_major_locator(MultipleLocator(25))
            pass
        if t1 == 'H3':
            ax.set_xlim(-1,8.1)
            ax.xaxis.set_major_locator(MultipleLocator(2))
            pass
        if t2 == 'H3':
            ax.set_ylim(-1,8.1)
            ax.yaxis.set_major_locator(MultipleLocator(2))
            pass
        if t1 == 'SF6':
            ax.set_xlim(-0.25,3)
            ax.xaxis.set_major_locator(MultipleLocator(1))
        if t2 == 'SF6':
            ax.set_ylim(-0.25,3)
            ax.yaxis.set_major_locator(MultipleLocator(1))
             
        #ax.set_xlabel(ylabs[t1], labelpad=0.8)
        ax.minorticks_on()
        
        if m == 0:
            #ax.set_ylabel(ylabs[t2], labelpad=0.2)
            ax.set_ylabel('{} ({})'.format(ylabs[t2],ylabs_units[t2]), labelpad=0.2)
            #ax.tick_params(axis='y', pad=0.005, rotation=45)
        else:
            #ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel('')
            
        if m == 1:
            #ax.set_xlabel(r'log$_{{10}}$ {}'.format(ylabs[t1]), labelpad=0.8, loc='right')
            ax.text(1.15, -0.35, '{} ({})'.format(ylabs[t1],ylabs_units[t1]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)    
            
                  
leg = axes[0,len(mod_names)-1].legend(loc='upper left', bbox_to_anchor=(0.82, 1.0), frameon=False, handletextpad=0.01)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.savefig('./figures/rtd_mc_young.png', dpi=300)
plt.savefig('./figures/rtd_mc_young.svg', format='svg')
plt.show()






#
#
# Versus He4
#
#
tlist = ['H3','SF6','CFC12']

ms = ['o','D', '<']
ms = ['o','D', 'P']
fill = ['y','white', 'black']

# Pick RTD models to plot
# reconfigure the list
mod_list    = [1,2,3,9] # Order to plot in
mod_names   = ['pfm','emm','epm','epm-epm'] 

mask = [i in mod_names for i in mod_df_master['mod'].to_list()]

mod_df_ = mod_df_master.copy()[mask]
mod_df_.reset_index(drop=True, inplace=True)


fig, axes = plt.subplots(nrows=3, ncols=len(mod_names), figsize=(8, 6))
fig.subplots_adjust(top=0.92, bottom=0.1, left=0.08, right=0.88, hspace=0.5, wspace=0.12)
for k in range(len(tlist)):

    # Plot tracer-tracer scatter plots
    #t1 = cmb[k][0]
    #t2 = cmb[k][1]
    t1  = 'He4_ter'
    t2  = tlist[k]
    
    # Plot individual models seperate
    for m in range(len(mod_names)):        
        ax = axes[k,m]
        
        mm = mod_names[m]
        dd = copy.deepcopy(mod_df_)[mod_df_['mod'] == mm]
        
        # plot just for single rtd
        # Plot pfm and emm as lines - not scatter points
        if mm in ['pfm','emm']:
            # plot just for single rtd
            #ax.plot(dd[t1], dd[t2], marker='None', color='C{}'.format(m), linewidth=2.0, alpha=1.0, zorder=5)
            ax.plot(dd[t1]*1.e-8, dd[t2], marker='None', color='C{}'.format(m), linewidth=2.0, alpha=1.0, zorder=5)
            # add age labels
            # Pain
            if t2 == 'H3':
                rt_pfm = [10,30,50,60,70]
            elif t2 == 'SF6':
                rt_pfm = [10,20,30,40,50,70]
            else:
                rt_pfm = [1, 10, 20, 30, 40, 50, 60, 70]
            rt_emm = [1, 10, 100, 250, 1000, 10000]
            if mm == 'pfm':
                [ax.annotate(i, xy=(pull_lab_xy(i,dd,t1)*1.e-8,pull_lab_xy(i,dd,t2)), xycoords='data', color='black', fontsize=10, 
                         horizontalalignment='center', verticalalignment='center', zorder=5) for i in rt_pfm]
            if mm == 'emm':
                [ax.annotate(i, xy=(pull_lab_xy(i,dd,t1)*1.e-8,pull_lab_xy(i,dd,t2)), xycoords='data', color='black', fontsize=10, 
                         horizontalalignment='center', verticalalignment='center', zorder=5) for i in rt_emm]
        else:
            ax.scatter(dd[t1]*1.e-8, dd[t2], marker='.', color='C{}'.format(m), alpha=0.4, zorder=5)    
    
        # plot all points
        ax.scatter(mod_df_[t1]*1.e-8, mod_df_[t2], marker='.', color='grey', alpha=0.1)
    
        if k == 0:
            try:
                mm_ = mm.split('-')
                ax.set_title('{}-{}'.format(name_map[mm_[0]], name_map[mm_[1]]))
            except IndexError:
                ax.set_title(name_map[mm])
        #
        # Observations
        for w in range(len(wells)):
            xx = copy.deepcopy(ens_dict)[t1][wells[w]].to_numpy().ravel()#[:20000]
            yy = copy.deepcopy(ens_dict)[t2][wells[w]].to_numpy().ravel()#[:20000]
            #sns.kdeplot(x=xx,y=yy, ax=ax, cmap=cmlist[w], thresh=0.05, zorder=3, shade=True)
            if t1 == 'He4_ter':
                #xx *= 1.e8
                ax.scatter(xx.mean(), yy.mean(), color='black', marker=ms[w], s=50.0, alpha=1.0, zorder=8, 
                           facecolors=fill[w], edgecolors='black', linewidth=1.5, label=wells[w].split('_')[0]) # no log use this
                #confidence_ellipse(xx, yy, ax, edgecolor='black', alpha=0.8, zorder=10)
                ax.errorbar(xx.mean(), yy.mean(), xerr=xx.std()*3, yerr=yy.std()*3, alpha=0.6, ecolor='black',zorder=6)
            elif t2 == 'He4_ter':
                #yy *= 1.e8
                ax.scatter(xx.mean(), yy.mean(), color='black', marker=ms[w], s=50.0, alpha=1.0, zorder=8, 
                           facecolors=fill[w], edgecolors='black', linewidth=1.5, label=wells[w].split('_')[0]) # no log use this
                #confidence_ellipse(xx, yy, ax, edgecolor='black', alpha=0.8, zorder=10)
                ax.errorbar(xx.mean(), yy.mean(), xerr=xx.std()*3, yerr=yy.std()*3, alpha=0.6, ecolor='black')
            else:
                ax.scatter(xx.mean(), yy.mean(), color='black', marker=ms[w], s=50.0, alpha=1.0, zorder=8, 
                           facecolors=fill[w], edgecolors='black', linewidth=1.5, label=wells[w].split('_')[0]) # no log use this
                #confidence_ellipse(xx, yy, ax, edgecolor='black', alpha=0.8, zorder=10)
                ax.errorbar(xx.mean(), yy.mean(), xerr=xx.std()*3, yerr=yy.std()*3, alpha=0.6, ecolor='black',zorder=6)         
        # Clean-up
        #ax.set_yscale('log')
        ax.set_xscale('log')
        if t1 == 'CFC12':
            ax.set_xlim(-2,60)
            ax.xaxis.set_major_locator(MultipleLocator(25))
            pass
        if t2 == 'CFC12':
            ax.set_ylim(-2,60)
            ax.yaxis.set_major_locator(MultipleLocator(25))
            pass
        if t1 == 'H3':
            ax.set_xlim(-0.25,8.1)
            ax.xaxis.set_major_locator(MultipleLocator(2))
            pass
        if t2 == 'H3':
            ax.set_ylim(-0.25,8.1)
            ax.yaxis.set_major_locator(MultipleLocator(2))
        if t1 == 'He4_ter':
            xticks = [1.e-10, 1.e-9, 1.e-8, 1.e-7, 1.e-6]
            ax.set_xticks(ticks=xticks, labels=np.log10(xticks).astype(int))
            ax.set_xlim(2.e-10, 1.e-6)
            #ax.xaxis.set_major_locator(ticker.LogLocator(numticks=8))
            #ax.xaxis.set_major_locator(MultipleLocator(3))
            pass
        if t2 == 'He4_ter':
            #ax.set_ylim(-1,12.1)
            #ax.yaxis.set_major_locator(MultipleLocator(3))
            pass
        if t1 == 'SF6':
            ax.set_xlim(-0.25,3)
            ax.xaxis.set_major_locator(MultipleLocator(1))
        if t2 == 'SF6':
            ax.set_ylim(-0.25,3)
            ax.yaxis.set_major_locator(MultipleLocator(1))
        
        ax.minorticks_on()
        
        if m == 1:
            #ax.set_xlabel(r'log$_{{10}}$ {}'.format(ylabs[t1]), labelpad=0.8, loc='right')
            ax.text(1.05, -0.35, r'log$_{{10}}$ {} ({})'.format(ylabs[t1],ylabs_units[t1]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        
        if m == 0:
            ax.set_ylabel('{} ({})'.format(ylabs[t2],ylabs_units[t2]), labelpad=0.2)
            #ax.tick_params(axis='y', pad=0.005, rotation=45)         
        else:
            #ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel('')
        
leg = axes[0,len(mod_names)-1].legend(loc='upper left', bbox_to_anchor=(0.82, 1.0), frameon=False, handletextpad=0.01)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.savefig('./figures/rtd_mc_old.png', dpi=300)
plt.savefig('./figures/rtd_mc_old.svg', format='svg')
plt.show()









#------------------------------
#
# Plot some RTDs
#
#------------------------------

import convolution_integral_utils as conv

tr_in = C_in_dict['CFC12'].copy()
conv_ = conv.tracer_conv_integral(C_t=tr_in, t_samp=tr_in.index[-1]) 

tau_bar = 20

conv_.update_pars(tau=tau_bar, mod_type='piston')
pfm_rtd = conv_.gen_g_tp()

conv_.update_pars(tau=tau_bar, mod_type='exponential')
emm_rtd = conv_.gen_g_tp()

conv_.update_pars(tau=tau_bar, mod_type='dispersion', D=0.5)
dmm_rtd = conv_.gen_g_tp()

taus = conv_.tau_list



# plot them
fig, ax = plt.subplots(figsize=(4.5,3))
#ax.plot(taus, pf_rtd,  ls='-',   color='black', label='piston-flow')
ax.axvline(tau_bar, ls='-', color='black',    linewidth=2.0, label='piston-flow')
ax.plot(taus, emm_rtd, ls='--', color='black', linewidth=2.0, label='exponential')
ax.plot(taus, dmm_rtd, ls=':', color='black', linewidth=2.0, label='dispersion')

ax.set_xlabel('Age (years)')
ax.set_ylabel('Fraction')
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.set_xlim(0,75)
ax.set_ylim(0.0, 0.1)

ax.legend()
fig.tight_layout()
plt.savefig('figures/rtd_curves.png',dpi=300)
plt.show()



"""


#----------------------------------
# Tracer-Tracer Consistency Plots
#----------------------------------

def pull_data(mod_name, tracer, well, numstd):
    tr = tracer
    # Find correct model results
    mod_df_  = (mod_df_master[mod_df_master['mod'] == mod_name])
    mod_df   = mod_df_.copy().loc[:,tr]
    
    # Build similiar observation dataframe
    obs = []
    for t in tr:
        obs_ = copy.deepcopy(ens_dict)[t][well]
        if t == 'He4_ter':
            obs_ *= 1.e8
        obs.append(obs_)
    obs = pd.concat(obs,axis=1)
    
    # Calculate parameters that fall within observation error bounds
    res_ = []
    for t in tr:
        #perr = 0.1
        #r  = np.where((mod_df[t] >= obs[t].min()*(1-perr)) & (mod_df[t] <= obs[t].max()*(1+perr)), True, False)
        low  = obs[t].mean() - obs[t].std()*numstd
        high = obs[t].mean() + obs[t].std()*numstd
        r  = np.where((mod_df[t] >= low) & (mod_df[t] <= high), True, False)
        res_.append(r)
    res = pd.DataFrame(np.column_stack(res_), columns=tr)
    res_comp = res.sum(axis=1) == len(tr)
    
    # Parameters that can explain tracer observations
    pars_ = mod_df_.loc[:,['f1', 'f2', 'tau1', 'tau2', 'tau', 'eta']+tr]
    pars  = pars_[res_comp.values.ravel()]
    return pars

#p1 = pull_data('epm', ['He4_ter'], 'PLM1', 4)


# Can any models explain all three  tracers
tracers  = ['He4_ter', 'CFC12','H3']
well     = 'PLM1'
sdnum    = 4
model    = 'emm-pfm'

mt = pull_data(model, tracers, well, sdnum)




#
# Plots
#

tracers  = ['He4_ter', 'CFC12','H3']
well     = 'PLM1'
sdnum = 3


#
# EMM model
m  = 'emm'
fig, ax = plt.subplots()
for i,t in zip(range(len(tracers)), tracers):
    p = pull_data(m, [t], well, sdnum)
    #ax.scatter(p['tau1'], p['eta'], marker='o', edgecolors='C{}'.format(i), facecolors='none', label=ylabs[t])
    sns.kdeplot(p['tau1'], ax=ax, color='C{}'.format(i), label=ylabs[t])
ax.set_xlabel(r'$\tau$ (years)')
#ax.set_ylabel(r'$\eta$ (-)')
ax.set_yticklabels([])
ax.set_yticks([])
ax.set_title(m)
ax.legend()
fig.tight_layout()
plt.show()


#
# EPM model
m  = 'epm'

fig, ax = plt.subplots()
for i,t in zip(range(len(tracers)), tracers):
    p = pull_data(m, [t], well, sdnum)
    ax.scatter(p['tau1'], p['eta'], marker='o', edgecolors='C{}'.format(i), facecolors='none', label=ylabs[t])
ax.set_xlabel(r'$\tau$ (years)')
ax.set_ylabel(r'$\eta$ (-)')
ax.set_title(m)
ax.legend()
fig.tight_layout()
plt.show()


#
# EMM-PFM model
m  = 'emm-pfm'
pp = ['tau1','tau2','f1']
cmb = list(combinations(pp, 2))
labs = {'tau1':r'$\tau_{1}$ (years)',
        'tau2':r'$\tau_{2}$ (years)',
        'tau':r'$\tau$ (years)',
        'f1':'$f_{1}$',
        'f2':'$f_{2}$'}

fig, ax = plt.subplots(nrows=len(cmb)+1,ncols=1, figsize=(4,10))
for i,t in zip(range(len(tracers)), tracers):
    p = pull_data(m, [t], well, sdnum)
    for j in range(len(cmb)+1):
        if j < len(cmb):
            ax[j].scatter(p[cmb[j][0]], p[cmb[j][1]], marker='o', edgecolors='C{}'.format(i), facecolors='none', label=ylabs[t])
            ax[j].set_xlabel(labs[cmb[j][0]])
            ax[j].set_ylabel(labs[cmb[j][1]])
        if j == len(cmb):
            ax[j].hist(p['tau'], histtype='step', color='C{}'.format(i), bins=20)
            #sns.kdeplot(p['tau'], ax=ax[j], bw_adjust=0.5, color='C{}'.format(i), label=ylabs[t])
            ax[j].set_xlabel(labs['tau'])
            #ax[j].set_yticklabels([])
            #ax[j].set_yticks([])
ax[0].set_title(m)
#ax[0].legend()
fig.tight_layout()
plt.show()


#
# EMM-EMM model
m  = 'emm-emm'
fig, ax = plt.subplots(nrows=len(cmb)+1,ncols=1, figsize=(4,10))
for i,t in zip(range(len(tracers)), tracers):
    p = pull_data(m, [t], well, sdnum)
    for j in range(len(cmb)+1):
        if j < len(cmb):
            ax[j].scatter(p[cmb[j][0]], p[cmb[j][1]], marker='o', edgecolors='C{}'.format(i), facecolors='none', label=ylabs[t])
            ax[j].set_xlabel(labs[cmb[j][0]])
            ax[j].set_ylabel(labs[cmb[j][1]])
        if j == len(cmb):
            ax[j].hist(p['tau'], density=True, histtype='step', color='C{}'.format(i), bins=20)
            #sns.kdeplot(p['tau'], ax=ax[j], bw_adjust=0.5, color='C{}'.format(i), label=ylabs[t])
            ax[j].set_yticks([])
            ax[j].set_yticklabels([])
            ax[j].set_yticks([])
ax[0].set_title(m)
#ax[0].legend()
fig.tight_layout()
plt.show()


#
# PFM-PFM model
m  = 'pfm-pfm'
fig, ax = plt.subplots(nrows=len(cmb)+1,ncols=1, figsize=(4,10))
for i,t in zip(range(len(tracers)), tracers):
    p = pull_data(m, [t], well, sdnum)
    for j in range(len(cmb)+1):
        if j < len(cmb):
            ax[j].scatter(p[cmb[j][0]], p[cmb[j][1]], marker='o', edgecolors='C{}'.format(i), facecolors='none', label=ylabs[t])
            ax[j].set_xlabel(labs[cmb[j][0]])
            ax[j].set_ylabel(labs[cmb[j][1]])
        if j == len(cmb):
            ax[j].hist(p['tau'], density=True, histtype='step', color='C{}'.format(i), bins=20)
            #sns.kdeplot(p['tau'], ax=ax[j], bw_adjust=0.5, color='C{}'.format(i), label=ylabs[t])
            ax[j].set_yticks([])
            ax[j].set_yticklabels([])
            ax[j].set_yticks([])
ax[0].set_title(m)
#ax[0].legend()
fig.tight_layout()
plt.show()
"""
