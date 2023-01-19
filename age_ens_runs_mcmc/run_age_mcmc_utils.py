#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 12:42:46 2021

@author: nicholasthiros
"""

import numpy as np
import pandas as pd
import pickle
import sys
import scipy
import datetime
import pdb
import os
#import lmfit as lm
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, MaxNLocator, LogLocator)
import matplotlib.ticker as ticker
plt.rcParams['font.size']=14
#plt.rcParams["text.usetex"] = False

sys.path.insert(0, '../utils')
import convolution_integral_utils as conv
import noble_gas_utils as ng_utils


import pymc3 as mc
import theano
import theano.tensor as TT
#from theano.compile.ops import as_op
import arviz as az


import copy


#-------------------------------------------------------------------
#
# Tau Posterior Analysis w/ MCMC
#
#-------------------------------------------------------------------

class ForwardMod(TT.Op):
    '''Wrapper to allow theano to work with convolution external code'''
    itypes = [TT.dvector] 
    otypes = [TT.dscalar]

    def __init__(self, conv_kwgs, par_names, tracer):
        kwargs = conv_kwgs.copy()
        self.C_t        = kwargs.get('C_t',       False)
        
        self.mod_type1  = kwargs.get('mod_type1', False)
        self.mod_type2  = kwargs.get('mod_type2', False)
        
        self.t_half     = kwargs.get('t_half',    False)
        self.rad_accum  = kwargs.get('rad_accum', False)
        self.J          = kwargs.get('J',         False) # included as uncertain parameter now
        self.eta        = kwargs.get('eta',       False) # included as uncertain parameter now
        self.D          = kwargs.get('D',         False) # included as uncertain parameter now
        self.bbar       = kwargs.get('bbar',      False)   
        self.Phi_im     = kwargs.get('Phi_im',    False)
        
        self.t = tracer
        
        # Parameter names that are variable in calibration
        self.p_names    = par_names
        
        # Initialize a parameter dictionary - needs to include all possible uncertain parameters
        self.p_dict     = {'tau1':False, 'tau2':0.0,
                           'f1':1.0,     'f2':0.0,
                           'eta1':False, 'eta2':False,
                           'D1':False,   'D2':False,
                           'J':False,
                           'thalf_cfc':False,
                           'lamsf6':False}
        
    def perform(self, node, inputs, outputs):
        """Forward Model"""
        #pdb.set_trace()
        #tau1, tau2, f1, f2 = inputs[0]
        
        # update parameter dict
        for i in range(len(self.p_names)):
            self.p_dict[self.p_names[i]] = inputs[0][i]
                
        if self.t=='He4_ter' and self.p_dict['J']==False:
            self.p_dict['J'] = np.log10(ng_utils.J_flux(Del=1., rho_r=2700, rho_w=1000, U=3.0, Th=10.0, phi=0.05))
        
        # Initialize Convolution Model
        conv_mod = conv.tracer_conv_integral(self.C_t.copy(), self.C_t.copy().index[-1])
        
        # First RTD
        conv_mod.update_pars(tau=self.p_dict['tau1'],
                             mod_type=self.mod_type1,
                             t_half=self.t_half,
                             rad_accum=self.rad_accum,
                             J=10**self.p_dict['J'],
                             eta=self.p_dict['eta1'],
                             D=self.p_dict['D1'],
                             bbar=self.bbar,
                             Phi_im=self.Phi_im)
        # First order decay of CFC
        if 'thalf_cfc' in self.p_names and self.t=='CFC12':
            conv_mod.t_half = self.p_dict['thalf_cfc']
            conv_mod.thalf_2_lambda(self.p_dict['thalf_cfc'])
        # Using He3 and H3 seperate tracers -- not the ratio
        #if self.rad_accum == '3He':
 	    #    # 3He concentration
        #    He3 =  conv_mod.convolve()
        #    # Tritium concentration
        #    conv_mod.rad_accum = 'False'
        #    H3 =  conv_mod.convolve()
        #    # Ratio
        #    cout1 = H3/He3
        #else:
        #    cout1 =  conv_mod.convolve()
        cout1 = conv_mod.convolve()
        
        
        # Second RTD -- if considering a BMM, else skip
        cout2 = 0.0 # initialize to zero
        if self.mod_type2:
            conv_mod2 = conv.tracer_conv_integral(self.C_t.copy(), self.C_t.copy().index[-1])
            conv_mod2.update_pars(tau=self.p_dict['tau2'],
                                 mod_type=self.mod_type2,
                                 t_half=self.t_half,
                                 rad_accum=self.rad_accum,
                                 J=10**self.p_dict['J'],
                                 eta=self.p_dict['eta2'],
                                 D=self.p_dict['D2'],
                                 bbar=self.bbar,
                                 Phi_im=self.Phi_im)
            # First order decay of CFC
            if 'thalf_cfc' in self.p_names and self.t=='CFC12':
                conv_mod2.t_half = self.p_dict['thalf_cfc']
                conv_mod2.thalf_2_lambda(self.p_dict['thalf_cfc'])
            #if self.rad_accum == '3He':
     	    #    # 3He concentration
            #    He3 =  conv_mod2.convolve()
            #    # Tritium concentration
            #    conv_mod2.rad_accum = 'False'
            #    H3 =  conv_mod2.convolve()
            #    # Ratio
            #    cout2 = H3/He3
            #else:
            #    cout2 =  conv_mod2.convolve()
            cout2 =  conv_mod2.convolve()
        
        # Fraction weighted concenrations
        cout = self.p_dict['f1']*cout1 + self.p_dict['f2']*cout2
        
        # CFC and SF6 degradation and contamination corrections
        #if self.t == 'CFC12' and 'lamcfc' in self.p_dict:
        #    cout *= (1-self.p_dict['lamcfc'])
            
        if self.t == 'SF6' and 'lamsf6' in self.p_dict:
            cout *= (1+self.p_dict['lamsf6'])
        
        outputs[0][0] = np.array(cout)


class CFC_corr(TT.Op):
    '''Wrapper to allow theano to work with convolution external code'''
    # This does not work right now because trying to mix Theano with Numpy
    itypes = [TT.dscalar]
    otypes = [TT.dscalar]

    def __init__(self, cfc_mod):
        self.cfc_mod = cfc_mod       
        
    def perform(self, node, inputs, outputs):
        """Forward Model"""
        #pdb.set_trace()
        lam_cfc = inputs[0]
        cfc_corr = self.cfc_mod * (1-lam_cfc) # Applying degredation to the model
        outputs[0][0] = np.array(cfc_corr)



#---------------------------
#
# MCMC Functions
#
#---------------------------
class conv_mcmc():
    def __init__(self, well, tracer, obs_kwgs, conv_kwgs, prior_kwgs, savedir, savenum):
        self.well      = well
        self.tracer    = tracer
        
        # unpack parameter priors
        self.tau1_low   = prior_kwgs.get('tau1_low',  False)
        self.tau1_high  = prior_kwgs.get('tau1_high', False)
        self.f1_low     = prior_kwgs.get('f1_low',    False)
        self.f1_high    = prior_kwgs.get('f1_high',   False)
        
        self.tau2_low   = prior_kwgs.get('tau2_low',  False)
        self.tau2_high  = prior_kwgs.get('tau2_high', False)
        
        self.eta1_low   = prior_kwgs.get('eta1_low',  False)
        self.eta1_high  = prior_kwgs.get('eta1_high', False)
        self.eta2_low   = prior_kwgs.get('eta2_low',  False)
        self.eta2_high  = prior_kwgs.get('eta2_high', False)
        
        self.D1_low    = prior_kwgs.get('D1_low',  False)
        self.D1_high   = prior_kwgs.get('D1_high', False)
        self.D2_low    = prior_kwgs.get('D2_low',  False)
        self.D2_high   = prior_kwgs.get('D2_high', False)
        
        self.J_mu      = prior_kwgs.get('J_mu', False)
        self.J_sd      = prior_kwgs.get('J_sd', False) 
        
        self.cfc_thalf_lo  = prior_kwgs.get('cfc_thalf_lo', False) # cfc decay half-lives in years, see Hinsby 2007 WRR
        self.cfc_thalf_hi  = prior_kwgs.get('cfc_thalf_hi', False)
        
        #self.lamcfc    =  prior_kwgs.get('lamcfc', False) # True or False Flag
        
        self.par_names   = prior_kwgs.get('par_names', False)
        
        
        # observation info
        self.obs = obs_kwgs.copy()
        
        # convolution info
        self.conv_kwgs = conv_kwgs.copy()
        self.mod_type1 = conv_kwgs.get('mod_type1', False)
        self.mod_type2 = conv_kwgs.get('mod_type2', False)
        
        # setup directory to save traces
        self.setup_dirs(savedir, savenum)
        
        # mcmc output tracers
        self.idata = None
        
        # prior predictive
        self.prior_pred = None
                               
        
    def setup_dirs(self, savedir, savenum):
        '''setup directories to save traces'''
        #pdb.set_trace()
        # Make new directory
        '' if os.path.exists(savedir) and os.path.isdir(savedir) else os.makedirs(savedir)
        
        tracer_list = '.'.join(self.tracer)
        
        if self.mod_type2:
            mod_type = '{}-{}'.format(self.mod_type1, self.mod_type2)
        else:    
            mod_type = self.mod_type1
             
        self.trace_name = './{}/{}.{}.{}.{}.netcdf'.format(savedir, self.well, tracer_list, mod_type, savenum)  
        self.trace_csv  = './{}/{}.{}.{}.{}.csv'.format(savedir,    self.well, tracer_list, mod_type, savenum)                      
        
        # Figures directory
        # ex dir: ./conv_figs/PLM1.CFC12.piston.0
        #self.idir  = './conv_figs/{}.{}.{}.{}'.format(well, tracer, conv_kwgs['mod_type'], savenum)
        #'' if os.path.exists(self.idir) and os.path.isdir(self.idir) else os.makedirs(self.idir)
    
                                    
    def frombeta(self, betaval, low, high):
        '''scales beta dist [0,1] to interval [low,high]'''
        return betaval * (high-low)+low


    def fromlog(self, logval):
        '''converts from beta dist. value to tau in years'''
        return 10**logval

    
    
    def build_mcmc_model_joint_sT(self):
        '''Using Student T likelihood'''
        model = mc.Model()
        with model:
            #pdb.set_trace()
            #--- 
            # Priors
            #---
            prior_conv_dict = {}
            
            # mean residence time or residence time of young fraction if bmm
            tau1 = mc.Uniform('tau1', self.tau1_low, self.tau1_high)
            prior_conv_dict['tau1'] = tau1
            
            # student-t likelikihood degrees of freedom
            #nu_dfs = mc.Uniform('nu', 29.0, 30.0)
            nu_dfs_ = mc.Beta('nu_', alpha=2.0, beta=0.1) # in interval [0,1]
            nu_dfs  = mc.Deterministic('nu', nu_dfs_*(30.0-5.0)+5.0) # degrees of freedom now between [1,30]
            
            
            if 'J' in self.par_names:
                #J  = mc.Normal('J', mu=np.log10(3.30e-11), sigma=0.33)
                #J  = mc.Normal('J', mu=np.log10(3.30e-11), sigma=0.005)
                J   = mc.Normal('J', mu=self.J_mu, sigma=self.J_sd)
                prior_conv_dict['J'] = J
            
            if self.mod_type2: # activates BMM parameters
                tau2 = mc.Uniform('tau2', self.tau2_low, self.tau2_high)
                f1   = mc.Uniform('f1',   self.f1_low,   self.f1_high)
                f2   = mc.Deterministic('f2', (1-f1))
                tau  = mc.Deterministic('tau', f1*(tau1)+f2*(tau2))
                
                prior_conv_dict['tau2'] = tau2
                prior_conv_dict['f1']   = f1
                prior_conv_dict['f2']   = f2
            
            if self.mod_type1 == 'exp_pist_flow': 
                eta1  = mc.Uniform('eta1', self.eta1_low, self.eta1_high)
                prior_conv_dict['eta1'] = eta1

            if self.mod_type2 == 'exp_pist_flow': 
                eta2  = mc.Uniform('eta2', self.eta2_low, self.eta2_high)
                prior_conv_dict['eta2'] = eta2
            
            if self.mod_type1 == 'dispersion': 
                D1  = mc.Uniform('D1', self.D1_low, self.D1_high)
                prior_conv_dict['D1'] = D1
                
            if self.mod_type2 == 'dispersion': 
                D2  = mc.Uniform('D2', self.D2_low, self.D2_high)
                prior_conv_dict['D2'] = D2
            
            # Decay correction factors for CFC 
            if 'thalf_cfc' in self.par_names:
                # cfc_obs = cfc_mod * (1-lam); lam [0,1] -- lam=0 is no change. lam=1 is 100% decrease
                #BoundedNormal = mc.Bound(mc.HalfNormal, upper=1.0)
                #lamcfc = BoundedNormal('lamcfc', sigma=0.5/3)
                #prior_conv_dict['lamcfc'] = lamcfc
                # First order (Radioactive-like) decay of CFC12 -- half-lifes from Hinby 2007
                thalf_cfc_ = mc.Beta('thalf_cfc_', alpha=2.0, beta=2.0) # in interval [0,1]
                thalf_cfc  = mc.Deterministic('thalf_cfc', thalf_cfc_*(self.cfc_thalf_hi -self.cfc_thalf_lo)+self.cfc_thalf_lo)
                prior_conv_dict['thalf_cfc'] = thalf_cfc
                
            
            # Contamination correction factors for SF6 
            if 'lamsf6' in self.par_names:
                # sf6_obs = sf6_mod * (1+lam); lam [0,1] -- lam=0 is no change. lam=1 is 100% increase
                #lamsf6 = mc.HalfNormal('lamsf6', sigma=0.5/3) # 3sigma within 0.5 of mean
                lamsf6 = mc.HalfNormal('lamsf6', sigma=0.5/3) # 3sigma within 0.5 of mean
                prior_conv_dict['lamsf6'] = lamsf6          

            
            
            #---
            # Observations
            #---
            # NET - 01/23/22 observation ensemble obs_df now includes analytical error
            #pdb.set_trace()
            obs_mu    = np.array([self.obs[w]['obs_df'].to_numpy().mean() for w in self.tracer])
            obs_nerr  = np.array([self.obs[w]['obs_df'].to_numpy().std() for w in self.tracer]) # includes ng and analytical errors
            obs_perr  = obs_mu * np.array([self.obs[w]['obs_perr'] for w in self.tracer])       # pure analytical error
            obs_err   = obs_nerr + obs_perr

            # build observation error covariance matrix for joint tracer inversions
            #obs_ens_joint = np.array([self.obs[w]['obs_df'].to_numpy().ravel()+1.e-20 for w in self.tracer])            
            #obs_cov = np.cov(obs_ens_joint)
 
    
    
            #--- 
            # Convolution -- Forward Model
            #---
            #pdb.set_trace()
            mod_out = []
           
            #par_names = ['tau1','tau2','f1','f2'] 
            #theta     = TT.as_tensor_variable([tau1,tau2,f1,f2])
            
            theta_ = [prior_conv_dict[i] for i in self.par_names]
            theta  = TT.as_tensor_variable(theta_)
           
            for w in self.tracer:
                self.conv_kwgs[w]['mod_type1'] = self.mod_type1
                self.conv_kwgs[w]['mod_type2'] = self.mod_type2
                conv_kwgs = copy.deepcopy(self.conv_kwgs[w])
                mod  = ForwardMod(conv_kwgs, self.par_names, w)  
                mod_ = mod(theta)
                mod_out.append(mod_)       
            mod_out = TT.as_tensor_variable(mod_out)
            
            
            #---
            # Likelihood -- assuming normal observation error model
            if len(self.tracer) == 1:
                #like = mc.Normal('like', mu=mod_out, sd=obs_err, observed=obs_mu)
                like = mc.StudentT('like', mu=mod_out, sigma=obs_err, nu=nu_dfs, observed=obs_mu) 
                
            else:
            #    # Not sure the proper way to do it for joint concentrations 
            #    #like = mc.Normal('like', mu=mod_out, sd=obs_err, observed=obs_mu) # no off-diagonal terms 
            #    #like = mc.MvNormal('like', mu=mod_out, cov=obs_cov, shape=(len(self.tracer), len(self.tracer)))
                like = mc.StudentT('like', mu=mod_out, sd=obs_err, nu=nu_dfs, observed=obs_mu) 
        return model

    
    def sample_prior(self):
        with self.build_mcmc_model_joint_sT():   
            prior = mc.sample_prior_predictive(1000)
            self.prior_pred = prior
            return prior


    def sample_mcmc(self):
        #with self.build_mcmc_model_joint(): 
        with self.build_mcmc_model_joint_sT(): 
            #prop_var = np.eye(par.shape[1]) * np.ones(par.shape[1])*0.1
            #prop_var = np.ones(par.shape[1])*0.25
            sampler = mc.DEMetropolisZ(tune_interval=1000)#, S=prop_var) #10000 works
            #sampler = mc.Metropolis()
            #sampler = mc.Slice()
            
            trace = mc.sample(tune=10000, draws=10000, step=sampler, chains=3, cores=3, random_seed=123423, 
                              idata_kwargs={'log_likelihood':False}, discard_tuned_samples=True, return_inferencedata=False)       
            
            if self.prior_pred:
                idata = az.from_pymc3(trace, prior=prior, log_likelihood=False)
            else:
                idata = az.from_pymc3(trace, log_likelihood=False)
            self.idata = idata
            # Save the full trace
            az.to_netcdf(idata, self.trace_name)
            # Save the posterior sampes
            #post_samps = np.asarray(idata.posterior['tau'])
            #np.savetxt(self.trace_csv, post_samps)
            return idata


    def plots(self):        
        with self.build_mcmc_model_joint_sT(): 
            idata = az.from_netcdf(self.trace_name)
            # Convert from beta parameters to normal
            #vrs = ['tau'] 
            #for p in vrs:
            #    idata.posterior[p] = idata.posterior[p] * (tau_high-tau_low)+tau_low
            #    idata.prior[p]     = idata.prior[p] * (tau_high-tau_low)+tau_low
         
            trace_summary = az.summary(idata, round_to=6)
            #print (trace_summary)
            
            # Traceplot
            az.plot_trace(idata)
            #plt.savefig(os.path.join(self.idir, 'trace_plot.jpg'), dpi=300)
            plt.tight_layout()
            plt.show()
            
            # Posterior plot
            #az.plot_posterior(idata, round_to=3)
            #plt.savefig(os.path.join(self.idir, 'post.jpg'), dpi=300)
            #plt.show()
            
            # Posterior with prior
            #ax = az.plot_density([idata.posterior, idata.prior],
            #                     data_labels=["Posterior", "Prior"],
            #                     shade=0.1,
            #                     hdi_prob=1.0,
            #                     textsize=14)
            #plt.savefig(os.path.join(self.idir, 'post_prior.jpg'), dpi=300)
            #plt.show()



# Moved all this to another script
"""
#------
# Read in obseration ensembles and input histories
map_dict   = pd.read_pickle('./map_dict.pk')
ens_dict   = pd.read_pickle('./ens_dict.pk')
C_in_dict  = pd.read_pickle('./C_in_dict.pk')

#-----
# MCMC inversion info
well      = 'PLM1'
tracers   = ['CFC12','SF6','He4_ter']


#----
# set up observations
okw  = {}
for tt in tracers:
    okw[tt] = {}
    okw[tt]['obs_map']  = map_dict[tt].loc[well,tt]
    okw[tt]['obs_df']   = ens_dict[tt][well]
    okw[tt]['obs_perr'] = 5.0


#----
# set up priors for tau
pkw = {}
pkw['tau_low']  =  0.1
pkw['tau_high'] =  10000.0


#----
# set up convolution hyperparameters
ckw = {}
for tt in tracers:
    ckw[tt] = {}
    ckw[tt]['mod_type']  = 'exponential'
    
    if tt in tracers:
        ckw[tt]['C_t']       = (C_in_dict[tt]).copy()
        ckw[tt]['t_half']    = False
        ckw[tt]['rad_accum'] = False
        ckw[tt]['J']         = False
        
    elif tt in ['He4_ter']:
        ckw[tt]['C_t']       = (C_in_dict[tt]*0.0).copy()
        ckw[tt]['t_half']    = False
        ckw[tt]['rad_accum'] = '4He'
        ckw[tt]['J']         = ng_utils.J_flux(Del=1., rho_r=2700, rho_w=1000, U=3.0, Th=10.0, phi=0.05)
        
    elif tt in ['H3_He3']:
        ckw[tt]['C_t']       = (C_in_dict.copy()['H3']).copy()
        ckw[tt]['t_half']    = 12.34
        ckw[tt]['rad_accum'] = '3He'
        ckw[tt]['J']         = False

        

## Runit
savedir   = 'conv_traces_joint'
savenum   = 0   # mostly just a counter for variable J
mc_conv = conv_mcmc(well, tracers, okw, ckw, pkw, savedir, savenum)
mc_conv.sample_mcmc()
#mc_conv.plots()
"""
