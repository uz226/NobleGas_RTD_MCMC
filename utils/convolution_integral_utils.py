# Update 08/25/2022
#   Matrix Diffusion RTD model now working



import pandas as pd
import numpy as np
import scipy 
import pdb
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy.integrate import cumtrapz
import datetime
from dateutil.relativedelta import relativedelta

from numba import jit



# Functions to run the Gardner 2016 Fracture/Matrix Diffusion RTD Model
# External so I can get speedups with the NUMBA Library
@jit(nopython=True)
def _trapz(y, x):
    '''Manual version of trapz integral that allows to be wrapped in numba'''
    return 0.5*((x[1:]-x[:-1])*(y[1:]+y[:-1])).sum()

@jit(nopython=True)
def _disp(_tadv, _tau, _D):
    '''Dispersion RTD
       _tadv is the integrating variable -- array of floats
       _tau is the mean residence time -- single value (parameter)
       _D is the longitudinal dispersion -- single value (parameter)
       Returns: an array of length len(_tadv)'''
    return ((1./_tau)/(np.sqrt(4.*np.pi*_D*(_tadv/_tau))))*(1./(_tadv/_tau))*np.exp(-1.*(((1.-(_tadv/_tau))**2)/(4.*_D*(_tadv/_tau)))) 

@jit(nopython=True)
def frac_rtd_numba_disp(tp, tau, D, bbar, kappa):
    '''Fracture/Matrix Diffusion RTD Model
       Assumes the advective component is the Dispersion RTD
         - tp: Array of residence times to loop over (years) -- set by the input function that is supplied
         - tau: Mean residence time (years) -- parameter
         - D: Longitudinal Dispersivity -- parameter
         - bbar: Mean fracture aperature (m) -- parameter
         - kappa: Calculated internal (below)
    '''
    #pdb.set_trace()
    f_t_tran = np.zeros_like(tp) # Holds the inner convolution integral values (RTD) from loop below
    for i in range(len(tp)): # Integrating up to the total travel time (tp)
        _tp = tp[i]
        # Calculate the advective time distribution up to the total transit time (tp)
        #tadv = np.logspace(-10, np.log10(_tp-1.e-10), 5000) # this discretization might be overkill... trying to get rid of the singularity...
        tadv = np.logspace(-6, np.log10(_tp-1.e-6), 1000)
        # The advective travel time disribution
        f_tadv = _disp(tadv, tau, D)
        # Immobile zone / retention transport time
        t_ret = _tp - tadv 
        Beta_tadv = tadv / bbar 
        f_ret_Beta_tadv = (kappa*Beta_tadv)/(2*np.sqrt(np.pi)*t_ret**(3/2))*np.exp((-1*kappa**2*Beta_tadv**2)/(4*t_ret)) # assumes uniform fracture aperature over the transport length.
        f_ret_Beta_tadv = f_ret_Beta_tadv/_trapz(f_ret_Beta_tadv[::-1],t_ret[::-1]) # normalize - this is key and was a big stumbling block - should it be published?
        # Combine advective and retention rtds
        f_i = f_ret_Beta_tadv*f_tadv  
        f_t_tran[i] = _trapz(f_i,tadv) # Scaler value
    return f_t_tran

@jit(nopython=True)
def frac_rtd_numba(tp, bbar, kappa, f_tadv_ext):
    '''Fracture/Matrix Diffusion RTD Model.
       Takes an externally provided advection RTD -- f_tadv_ext
         - tp: Array of residence times to loop over (years) -- set by the input function that is supplied
         - bbar: Mean fracture aperature (m) -- parameter
         - kappa: Calculated internal (below)
         - f_tadv_ext: Advective RTD distribution -- parameter 
             -- should sum to 1
             -- should be same length of input concentrations array
             -- first indices corresponds to tau=0, then counts up. Indices should match tp (input concentrations array)
                for now, this means it should be yearly timesteps, going from tau=0 at start to tau=len(tp) as last value
    '''
    #pdb.set_trace()
    if len(tp) != len(f_tadv_ext):
        print ('External advective function does not match the length of the tracer input function')
    f_t_tran = np.zeros_like(tp) # Holds the inner convolution integral values (RTD) from loop below
    for i in range(len(tp)): # Integrating up to the total travel time (tp)
        _tp = tp[i]
        # Calculate the advective time distribution up to the total transit time (tp)
        #tadv = np.logspace(-10, np.log10(tt-1.e-10), 10) # this discretization might be overkill... trying to get rid of the singularity...
        tadv = np.logspace(-6, np.log10(_tp-1.e-6), 1000)
        # The advective travel time disribution
        f_tadv = np.interp(tadv, tp[:i+1], f_tadv_ext[:i+1])
        # Immobile zone / retention transport time
        t_ret = _tp - tadv 
        Beta_tadv = tadv / bbar 
        f_ret_Beta_tadv = (kappa*Beta_tadv)/(2*np.sqrt(np.pi)*t_ret**(3/2))*np.exp((-1*kappa**2*Beta_tadv**2)/(4*t_ret)) # assumes uniform fracture aperature over the transport length.
        f_ret_Beta_tadv = f_ret_Beta_tadv / _trapz(f_ret_Beta_tadv[::-1],t_ret[::-1]) # normalize - this is key and was a big stumbling block - should it be published?
        # combine advective and retention rtds
        f_i = f_ret_Beta_tadv*f_tadv  
        f_t_tran[i] = _trapz(f_i,tadv) # Scaler value
    return f_t_tran






# Everything here is yearly resolution
class tracer_conv_integral():
    def __init__(self, C_t, t_samp):
        self.C_t    = C_t        # Time-series of tracer inputs, pandas dataframe indexed by date, only yearly resolution is working right now
        self.t_samp = t_samp     # Sample Date
        
        
    def update_pars(self, **kwargs):        
        # Required kwargs (at least once)
        self.tau        = kwargs.get('tau', None)
        self.mod_type   = kwargs.get('mod_type', None)
        
        # Optional kwargs for decaying tracers
        self.t_half     = kwargs.get('t_half', False)
        if self.t_half:
            self.thalf_2_lambda(self.t_half)
        else:
            self.lamba  = 0.0
            
        self.rad_accum  = kwargs.get('rad_accum', False)
        
        # Helium-4 production rates
        self.J          = kwargs.get('J', False)
      
        # Required if using 'exp_pist_flow'
        self.eta        = kwargs.get('eta', None)
         
        # Required if using 'dispersion' and 'frac_inf_diff'
        self.D          = kwargs.get('D', None)
        
        # Required for 'frac_inf_diff'
        self.bbar       = kwargs.get('bbar', None)   
        self.Phi_im     = kwargs.get('Phi_im', None)
        
        # SF6 decay
        self.J_sf6      = kwargs.get('J_sf6', None)
        
        # Adjective RTD for 'frac_inf_diff' model
        self.f_tadv_ext = kwargs.get('f_tadv_ext', None)
        
        
        
    def thalf_2_lambda(self, t_half):
        '''Convert half-life to decay constant
           Inputs
             -- t_half: half-life in years'''
        lamba = -1*np.log(0.5)/t_half
        self.lamba = lamba
        return lamba


    def gen_g_tp(self):
        ''' Generates one of the parametric age distributions as function of residence times, tp
          Inputs:
             -- tau: mean age in years (float)
             -- mod_type: age distribution (str)
             -- lamba: is decay the radioactive decay constant, lamda=0 for no decay (float)
             -- rad_accum: 'True' to accumulate tracer given half-life (str)
           Returns:
               -- Weighting Function, index 0 corresponds to sampling time, t_i
                  index[-1] is time 100*tau in the past
           '''
        #pdb.set_trace()
        # Array of residence times - everything in years right now
        tp = np.arange(0, len(self.C_t)).astype(float)
        tp[0] += 1e-5
        
        # Shift residence times in the event sampling date does not match the last date in the input series
        dtp = np.floor(self.t_samp - self.C_t.index[-1])
        tp += dtp
        self.tau_list = tp.copy()
            
        
        # Weighting Fucntions
        if self.mod_type == 'piston':
            g_tp = np.zeros(len(tp))
            ix = abs(tp-self.tau).argmin()
            g_tp[ix]=1.     
            
        if self.mod_type == 'exponential':
            g_tp = (1./self.tau)*np.exp(-tp/self.tau)          

        if self.mod_type == 'exp_pist_flow':
            eta = self.eta # eta is volume total/volume of exponential aquifer
            g_tp = np.zeros(len(tp))
            exp_mask = tp >= self.tau*(1-(1/eta))
            g_tp[exp_mask] = (eta/self.tau)*np.exp(-(eta*tp[exp_mask]/self.tau)+eta-1.)
            
        if self.mod_type == 'dispersion':
            # disp = dispersion parameter = 1/Pe = D/vx
            f1 = (1./self.tau)/(np.sqrt(4.*np.pi*self.D*(tp/self.tau)))
            f2 = (1./(tp/self.tau))*np.exp(-1.*(((1.-(tp/self.tau))**2)/(4.*self.D*(tp/self.tau))))
            g_tp = f1*f2
    
    
        if self.mod_type == 'frac_inf_diff.mint':
            # This is Payton's orginal code
            # modified from painter 2008 and villermeax 1981
            # see Gardner 2015
            # Parameters: self.Phi_m, self.D, self.tau, self.bbar
            D_o      = (2.3e-9)*60*60*24*365 #diffusion coefficient in water m2/s (time is all in years...)
            Phi_im   = self.Phi_im #immobile zone porosity
            R_im     = 1 #immoble zone retardation for linear sorption
            kappa    = Phi_im*np.sqrt((D_o*Phi_im**2)*R_im)
            f_t_tran = np.zeros(len(tp.copy()))
            #Beta_bar = self.tau/self.bbar # not needed?
            for i in np.arange(len(tp)):
                #pdb.set_trace()
                t_tran = tp.copy()[i] # total travel time, scalar
                # calculate the advective time distribution up to the transit time distribution
                tadv = np.logspace(-6, np.log10(t_tran-1.e-6), 1000) # this discretization might be overkill... trying to get rid of the singularity...
                # the advective travel time disribution  - assumes a dispersive RTD for now
                f_tadv = ((1./self.tau)/(np.sqrt(4.*np.pi*self.D*(tadv/self.tau))))*(1./(tadv/self.tau))*np.exp(-1.*(((1.-(tadv/self.tau))**2)/(4.*self.D*(tadv/self.tau)))) 
                # immobile zone / retention transport time
                t_ret = t_tran - tadv 
                Beta_tadv = tadv / self.bbar 
                f_ret_Beta_tadv = (kappa*Beta_tadv)/(2*np.sqrt(np.pi)*t_ret**(3/2))*np.exp((-1*kappa**2*Beta_tadv**2)/(4*t_ret)) # assumes uniform fracture aperature over the transport length.
                f_ret_Beta_tadv = f_ret_Beta_tadv / trapz(f_ret_Beta_tadv[::-1],t_ret[::-1]) # normalize - this is key and was a big stumbling block - should it be published?
                f_i = f_ret_Beta_tadv*f_tadv  
                f_t_tran[i] = trapz(f_i,tadv) # scalar 
            # for interpolation reasons...
            tp_ = tp.copy()
            f_t_tran[0]=0
            tp_[0]=0
            f_t_tran = f_t_tran / trapz(f_t_tran,tp_) # normalize distribution.
            # resample at t_prime:
            f2 = interp1d(tp_, f_t_tran, kind="linear")
            g_tp = f2(tp_)
            #pdb.set_trace()
            mean_travel_time = trapz(g_tp*tp_,tp_)/trapz(g_tp,tp_)
            self.FM_mu = mean_travel_time
            print ("mean travel time is " + '%3.2f' %mean_travel_time)
    

        if self.mod_type == 'frac_inf_diff':
            # Now using functions defined outside class and NUMBA for speedups
            # modified from painter 2008 and villermeax 1981
            # see Gardner 2015
            
            #pdb.set_trace()
            D_o      = (2.3e-9)*60*60*24*365 # diffusion coefficient in water m2/s (time is all in years...)
            Phi_im   = self.Phi_im # immobile zone porosity
            R_im     = 1 # immoble zone retardation for linear sorption
            kappa    = Phi_im*np.sqrt((D_o*Phi_im**2)*R_im)
            
            # Call the numba version...           
            try:
                len(self.f_tadv_ext) # want to use external advective RTD - for example, from particle tracking
                f_t_tran = frac_rtd_numba(tp, self.bbar, kappa, self.f_tadv_ext)
            except TypeError:
                f_t_tran = frac_rtd_numba_disp(tp, self.tau, self.D, self.bbar, kappa) # Assumes Dispersion Advective RTD
            # For interpolation reasons...
            f_t_tran[0]=0
            tp_ = tp.copy()
            tp_[0]=0
            f_t_tran = f_t_tran / np.trapz(f_t_tran, tp_) # normalize distribution.
            # resample at t_prime -- note, this does not seem to make a difference
            f2 = interp1d(tp_, f_t_tran, kind="linear")
            g_tp = f2(tp_)
            g_tp = f_t_tran
            mean_travel_time = np.trapz(g_tp*tp_, tp_)/np.trapz(g_tp, tp_)
            self.FM_mu = mean_travel_time
            #print ("mean travel time is " + '%3.2f' %mean_travel_time)
        
            
        # Nomalization
        g_tp = g_tp / g_tp.sum() 
        
        ## Moved this to below function
        ## Add in exponential decay or accumulation 
        ## Set lamba=0 and rad_accum=False to ignore radioactive decay
        #if self.rad_accum == '3He': 
        #    g_tp_ = g_tp*(1-np.exp(-self.lamba*tp)) # Accumulate tritogenic He3
        #else:
        #    g_tp_ = g_tp*np.exp(-self.lamba*tp) # decay given lamba; no decay if lamba=0 or False
        #          
        #self.g_tp = g_tp_
        return g_tp


    def convolve(self, **kwargs):
        '''Perform convolution integral with dot product rather than np.convolve
           Input has to be single tracer
           Returns:
               -- Concentration value corresponding to last index of C_t
           Inputs:
               -- tau: mean residence time in years
               -- mod_type: 'piston' or 'exponential' 
               -- lamba: decay constant in yr^-1, use thalf_2_lambda above
               -- rad_accum: True in want to acculate He, False otherwise'''
        #pdb.set_trace()
        
        #self.gen_g_tp()

        # set kwarg g_tau=SomeArray: will use the provided g_tau for the RTD. g_tau should be output from gen_g_tp() above
        # do not set a kwarg to regenerate the RTD with provided info.
        self.g_tp = kwargs.get('g_tau', None)
        if self.g_tp is None:
            self.g_tp = self.gen_g_tp()
    
    
        # Add in exponential decay or accumulateion
        # Array of residence times - everything in years right now
        tp = np.arange(0, len(self.C_t)).astype(float)
        tp[0] += 1e-5
        # Shift residence times in the event sampling date does not match the last date in the input series
        dtp = np.floor(self.t_samp - self.C_t.index[-1])
        tp += dtp
        # Set lamba=0 and rad_accum=False to ignore radioactive decay
        if self.rad_accum == '3He': 
            self.g_tp = self.g_tp*(1-np.exp(-self.lamba*tp)) # Accumulate tritogenic He3
        else:
            self.g_tp = self.g_tp*np.exp(-self.lamba*tp) # decay given lamba; no decay if lamba=0 or False
   
    
        # Accumulate 4He within input series
        if self.rad_accum == '4He':
            #pdb.set_trace()
            try:
                C_t_ = self.C_t.copy()['He4_ter'] + self.C_t.copy()['He4_ter'].index * self.J
            except KeyError:
                C_t_ = self.C_t.copy() + self.C_t.copy().index * self.J
            except ValueError:
                C_t_ = self.C_t.copy()['He4_ter'] + self.C_t.copy()['He4_ter'].index * self.J
                
        # Terrigenic SF6 contamination -- zero order accumulation
        elif self.rad_accum == 'SF6':
            C_t_ = self.C_t.copy()['SF6'] + self.C_t.copy()['SF6'].index * self.J_sf6.copy()
        else:
            C_t_ = self.C_t.copy()
      
        # Convolution using dot product
        C_t_in = np.flip(C_t_.to_numpy().ravel())
        C_i = np.dot(C_t_in, self.g_tp)
        
        self.C_i = C_i # Concentation at observation time
        return C_i


    """
    def convolve_slow(self):
        '''Perform convolution integral
           Returns:
               -- Concentration value corresponding to last index of C_t
           Inputs:
               -- tau: mean residence time in years
               -- mod_type: 'piston' or 'exponential' 
               -- lamba: decay constant in yr^-1, use thalf_2_lambda above
               -- rad_accum: True in want to acculate He, False otherwise'''
        #pdb.set_trace()
        #self.gen_g_t(self.tau, self.mod_type, self.lamba, self.rad_accum)
        self.gen_g_t()
        
        # Accumulate some 4He within input series
        if self.rad_accum == '4He':
            pdb.set_trace()
            self.C_t['He4_ter'] = self.C_t['He4_ter'].index * self.J
            #self.C_t['He4_ter'].copy() += self.C_t.copy().index * self.J
        
        try:
            C_i = [np.convolve(self.C_t.iloc[:,i], self.g_tp, mode='valid')[0] for i in range(self.C_t.shape[1])] 
            C_i = np.array(C_i)
        except IndexError:
            C_i = np.convolve(self.C_t, self.g_tp, mode='valid') 
                    
        #if len(self.g_tp) > len(self.C_t):
        #    C_i = np.flip(C_i)
        
        self.C_i = C_i # just concentation at observation time
        #self.tau = tau
        return C_i
    """




    


