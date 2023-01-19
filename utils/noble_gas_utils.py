#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:13:30 2021

@author: nthiros
"""

# This is a library of noble gas functions written by Payton Gardner - University of Montana, Dept. of Geosciences
# Modified by Nick Thiros, June 2021

import numpy as np
import pandas as pd
import pickle
#import lmfit as lm
import scipy
import pdb

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import (MaxNLocator, MultipleLocator)


# Inversion moved to own script
#import pymc3 as mc
#import seaborn as sb
#import theano
#import theano.tensor as tt
##from theano import as_op
#from theano.compile.ops import as_op
#import os
#import arviz as az


# Dry air mixing ratios
# The atmospheric standard taken from 
# Porcelli et al 2002, Noble Gases in Geochemistry and Cosmochemistry, pg 3.
atm_std = { 'N2'   : 0.781,
            'O2'   : 0.209,
            'Ar'   : 9.34e-3,
            'CO2'  : 3.7e-4,
            'Ne'   : 1.818e-5,
            'He'   : 5.24e-6,
            'CH4'  : 1.5e-6,
            'Kr'   : 1.14e-6,
            'H2'   : 7e-7,
            'N2O'  : 3e-7,
            'CO'   : 1e-7,
            'Xe'   : 8.7e-8,
            'Rn'   : 6e-20,
            'He3'  : 5.24e-6*0.000140/100,
            'He4'  : 5.24e-6,
            'Ne20' : 1.818e-5*90.50/100,
            'Ne21' : 1.818e-5*0.268/100,
            'Ne22' : 1.818e-5*9.23/100,
            'Ar36' : 9.34e-3*0.3364/100,
            'Ar38' : 9.34e-3*0.0632/100,
            'Ar40' : 9.34e-3*99.60/100,
            'Kr78' : 1.14e-6*0.3469/100,
            'Kr80' : 1.14e-6*2.2571/100,
            'Kr82' : 1.14e-6*11.523/100,
            'Kr83' : 1.14e-6*11.477/100,
            'Kr84' : 1.14e-6*57.00/100,
            'Kr86' : 1.14e-6*17.398/100,
            'Xe124': 8.7e-8*0.0951/100,
            'Xe126': 8.7e-8*0.0887/100,
            'Xe128': 8.7e-8*1.919/100,
            'Xe129': 8.7e-8*26.44/100,
            'Xe130': 8.7e-8*4.070/100,
            'Xe131': 8.7e-8*21.22/100,
            'Xe132': 8.7e-8*26.89/100,
            'Xe134': 8.7e-8*10.430/100,
            'Xe136': 8.7e-8*8.857/100}  

class noble_gas_fun():
    def __init__(self, gases, E, T, Ae, F, P, S=0.0):
        self.gases = gases         # list of gases to consider -- ['He','Ne','Ar','Kr','Xe']
        self.E  = E                # recharge elevation in meters
        self.T  = T                # recharge temperature in C
        self.Ae = Ae               # initial volume of entrapped air in ccSTP/g
        self.F  = F                # fractionation factor for CE model, unitless
        self.S  = S                # salinity
        self.P  = self.parse_P(P)  # pressure in GPa, takes key-words or floats 
                                   # P='1atm' for sea level, 
                                   # P='lapse_rate' for P based on elevation and fixed lapse rate,
                                   # P=float for any other Pressure in GPa       
        self.obs_dict = None       # Dictionary of observations keyed by noble gas name
        self.err_dict = None       # Dictionary of percent errors (ex 5.0) keyed by noble gase name        


    def parse_P(self, P):
        '''Returns the atmospheric pressure in GPa.
           Uses keywords:
               '1atm' for sea level, 'lapse_rate' for elevation based pressure [m], or a float already in GPa'''         
        if P == '1atm':
            return  0.000101325
        elif P == 'lapse_rate':
            return self.lapse_rate()
        else:
            return P

       
    def lapse_rate(self):
        ''' 
        Returns:
            -- The atmospheric pressure in GPa 
        Inputs:
            -- Elevation in meters
        Lapse rate taken from Kip Solomon University of Utah. Pressure in GPa.  
        Assumes a pressure of 1 atm at sea level.'''
        
        P_da = ((1-.0065*self.E/288.15)**5.2561)*0.000101325; # GPa From Kip 
        return P_da
      
    

    def solubility(self, gas):
        """
        Returns:
            Solubility coefficient for the noble gas of interest in water with units GPa.
            Satisfies the equation p_i = K_i*x_i.
            --    x_i is the mol fraction of gas i in water.
            --    K_i is the salt corrected Henry's coefficient.
            --    p_i is the partial pressure of gas i in GPa.  
            If temperature is less the 65 C salting coefficients are included.        
        Inputs:
            --    gas are: "He", "Ne", "Ar", "Kr", or "Xe". 
            --    T is the temperature in degrees C.  
            --    S is the salinity in mol l^-1.           
        Note: 1 K(atm) = 55.6 Km(atm Kg/mol)
        Solubility and salting coefficients are calculated using date from 
        Porcelli, D.; Ballentine, C. J. & Wieler, R. (ed.) Noble Gases in Geochemistry and 
        Cosmochemistry Mineralogical Society of America , 2002, 47 pg 546
        """
        gas=gas[0:2]
        T_k = self.T + 273.15;
        # noble gas solubility coefficients
        sol_dict = {'He':[-0.00953,0.107722,0.001969,-0.043825],
                    'Ne':[-7.259,6.95,-1.3826,0.0538],
                    'Ar':[-9.52,8.83,-1.8959,0.0698],
                    'Kr':[-6.292,5.612,-0.8881,-0.0458],
                    'Xe':[-3.902,2.439,0.3863,-0.221]};
        # need a Stechnow coefficient dictionary for the other noble gases as well 
        # taken from ng book pg 544 from kennedy
        setch_dict = {'He':[-10.081,15.1068,4.8127],
                     'Ne':[-11.9556,18.4062,5.5464],
                     'Ar':[-10.6951,16.7513,4.9551],
                     'Kr':[-9.9787,15.7619,4.6181],
                     'Xe':[-14.5524,22.5255,6.7513]};
        G = setch_dict[gas];
        G1 = G[0];
        G2 = G[1];
        G3 = G[2];
        if self.T < 65.: # add Setchenow coefficient. note: Setchenow coefficicient are only valid up to 65C
            setch = G1 + (G2/(.01*T_k))+ (G3*np.log(.01*T_k))
            gamma = np.exp(self.S*setch)
            #print 'stechenow salinity units uknown need Kennedy 1982 to get a better understanding...'
        else:
            gamma = 1.
        
        # now calculate the solubility
        A  = sol_dict[gas];
        A0 = A[0]  # taken from ng book pg 545 from Crovetto 1981
        A1 = A[1];
        A2 = A[2];
        A3 = A[3];
        
        if gas == 'He':
            ln_F_He = A0 + (A1/(.001*T_k)) + (A2/(.001*T_k)**2) + (A3/(.001*T_k)**3)
            F = np.exp(ln_F_He);
            Frac_He_gas = 5.24e-6/9.31e-3;
            X_Ar_water = 1./(np.exp(sol_dict['Ar'][0] + (sol_dict['Ar'][1]/(.001*T_k))+(sol_dict['Ar'][2]/(.001*T_k)**2) + (sol_dict['Ar'][3]/(.001*T_k)**3)))*9.31e-3;
            X_he_water = F*Frac_He_gas*X_Ar_water
            K_h = 5.24e-6/X_he_water; # Henry's coefficient in GPa
            K = gamma*K_h; # Salinty effects added solubility coeffient in GPa
        else:
            ln_K = A0 + (A1/(.001*T_k))+(A2/(.001*T_k)**2) + (A3/(.001*T_k)**3);
            K_h = np.exp(ln_K); # Henry's coefficient in GPa
            K = gamma*K_h;  # Salinty effects added solubility coeffient in GPa
        return K
                  
             
                
    def vapor_pressure(self):
        '''Returns:
               -- P_vapor, the vapor pressure in GPa using the Antione equation.'''     
        if self.T<=99.0:
            A=8.07131;
            B=1730.63;
            C=233.426;
        else:
            A=8.14019;
            B=1810.94;
            C=244.485;
            
        P = 10**(A-(B/(C+self.T))) #mmHG
        P=P/760.*101325  #Pa
        P_vapor=P/1.0e9  #GPa
        return P_vapor
    
    
    def equil_conc(self):
        '''Noble gas equilibrium concentrations
        Returns:
            -- The atmospheric equilibrium concentration of noble gas in ccSTP/g.'''
        gas_dict = dict.fromkeys(self.gases,0.0)
        for gas in self.gases:
            K_i = self.solubility(gas) 
            p_i = atm_std[gas]*self.P
            x_i = p_i/K_i           # mol fraction of gas_i in water
            C_i = x_i*(22414./18.)  # unit conversion to ccSTP/g
            gas_dict[gas] = C_i
        return gas_dict
   
    
    def equil_conc_dry(self):
        '''Noble gas equilibrium concentrations, calculates dry atm pressure 
        Returns:
            -- The atmospheric equilibrium concentration of noble gas in ccSTP/g.'''
        #pdb.set_trace()
        p_vapor = self.vapor_pressure()
        gas_dict = dict.fromkeys(self.gases,0.0)
        for gas in self.gases:
            K_i = self.solubility(gas) 
            p_i = atm_std[gas] * (self.P-p_vapor)
            x_i = p_i/K_i           # mol fraction of gas_i in water
            if self.T < 0.0:
                C_i = -9999.0
            else:
                C_i = x_i*(22414./18.)  # unit conversion to ccSTP/g
            gas_dict[gas] = C_i
        return gas_dict


    def ce_exc(self, add_eq_conc):
        ''' Closed System Excess Air Model
        Returns:
            -- Concentration of gas [ccSTP/g] due to closed system equilibrium excess air addition.      
        Inputs:
            -- add_eq_conc = True --> returns excess air conc + equil conc
            -- add_eq_conc = False --> returns excess air conc only'''
        #C_eq = self.equil_conc()
        #pdb.set_trace()
        C_eq = self.equil_conc_dry()
        ce_exc_dict = dict.fromkeys(self.gases, 0.0)
        for gas in C_eq:
            z_i = atm_std[gas]
            C_ex = ((1-self.F)*self.Ae*z_i)/(1+((self.F*self.Ae*z_i)/C_eq[gas]))
            if add_eq_conc:
                ce_exc_dict[gas] = C_ex + C_eq[gas]
            else:
                ce_exc_dict[gas] = C_ex
        return ce_exc_dict
    

    def update_pars(self, T, Ae, F, E):
        self.T = T
        self.Ae = Ae
        self.F = F
        self.E = E
        self.P = self.lapse_rate()




def F_w(C_i, T, S, E, Ae, F):
    '''Fractionation Factors relative to Ar
       Returns:
           - Fractionation factor for gas_i (C_i/C_Ar)_sample/(C_i/C_Ar)_asw
       Inputs:
           - C_i: Dictionary of noble gas observations ccSTP/g
           - T:   Recharge Temperature in C
           - S:   Salinity in mol/l
           - E:   Recharge elevation in meters
           - Ae:  Excess air in ccSTP/g (0 for no excess air)
           - F:   Fractionation factor (1 for no fractionation)'''
           
    P = 'lapse_rate' # cacluation pressure from recharge elevation
    gases = list(C_i.keys())
    ng_ = noble_gas_fun(gases=gases, E=E, T=T, Ae=Ae, F=F, P=P)
    
    C_i_atm  = ng_.equil_conc() # Atmospheric Equilibrium Concentrations
    R_atm = C_i_atm.copy()
    for gas in list(R_atm.keys()):
        R_atm[gas] = C_i_atm[gas]/C_i_atm['Ar']
    
    R_smp = C_i.copy() # Measured noble gas Ratios
    for gas in list(R_smp.keys()):
        R_smp[gas] = C_i[gas]/C_i['Ar']
    
    F_w = R_atm.copy()
    for gas in list(F_w.keys()):
        F_w[gas] = R_smp[gas]/R_atm[gas]
    
    for k in list(F_w.keys()):
        F_w['F_'+k] = F_w.pop(k)
    return F_w



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
    
    

#------------------
# He Production
# Gardner (2011) Artesian Basin Paper
#G = 4.0 # He production rate in ucm3/m3/yr
#        # Assumes gam = 1.0, and average upper crust compostition
#G_ = G/1000/1000*1.e-6 # cc/g/yr


def J_flux(Del, rho_r, rho_w, U, Th, phi):
    '''pg. 648 of Noble Gas Book in Geochemistry and Cosmochemistry (Porcelli, 2002)
       Returns:
           - He Accumulation Rate (ccSTP/g_water/year) due to matrix only
       Inputs:
           - Del:   release factor (typical=1)
           - rho_r: aquifer material density
           - rho_w: water density
           - U:     aquifer Uranim concentration in ug/g (micrograms/gram)
           - Th:    aquifer Thorium concentratin in ug/g (micrograms/gram)
           - phi:   porosity'''
    PU  = 1.19e-13
    PTh = 2.88e-14
    return Del * rho_r/rho_w * (U*PU + Th*PTh) * ((1-phi)/phi)

#J = J_flux(Del=1., rho_r=2700, rho_w=1000, U=3.0, Th=17.0, phi=0.05)



class ng_parse():
    def __init__(self, obs_dict, Ae, F, E, T):
        self.obs_dict = obs_dict  # For a single well, keys: He, Ne, Ar, Kr, Xe, R/Ra, He3
        self.Ae       = Ae
        self.F        = F
        self.E        = E
        self.T        = T
        
        self.obs_dict_ = obs_dict.copy()

    def He_comps_dontuse(self, Rterr):
        He4_obs = self.obs_dict['4He']
        He3_obs = self.obs_dict['3He']
        #---
        # 4He mass balance
        ng_    = noble_gas_fun(gases=['He'], E=self.E, T=self.T, Ae=self.Ae, F=self.F, P='lapse_rate')
        He_atm = ng_.equil_conc()['He']
        He_exc = ng_.ce_exc(add_eq_conc=False)['He']
        He_mod = ng_.ce_exc(add_eq_conc=True)['He']
        He_ter = He4_obs - He_mod
        #He_del = 100*(He_ter/He_atm-1)
        He_del = 100*(He_ter/He_atm)
        self.obs_dict_['He4_atm']   = He_atm
        self.obs_dict_['He4_eq_ex'] = He_mod
        self.obs_dict_['He4_ter']   = He_ter 
        self.obs_dict_['He4_del']   = He_del
        #---
        # 3He mass balance -- Followin NG book pg.642
        Ratm = 1.384e-6  #3He/4He in atmosphere
        #Req  = 1.360e-6
        #Rterr = 2.e-8 # Cook Book
        
        S = 0.0 # Salinity
        CF = 4.021e14/(1-S)
        #He_trit  = He3_obs - Req*He_atm - Ratm*He_exc - Rterr*He_ter
        He_trit  = He3_obs - (He4_obs-He_ter)*Ratm + He_atm*Ratm*(1-0.983) - He_ter*Rterr
        TU = He_trit * CF
        self.obs_dict_['He3_tu'] = TU
        
        
    def He_comps(self, Rterr):
        He4_obs = self.obs_dict['He4']
        He3_obs = self.obs_dict['He3']
        #---
        # 4He mass balance
        ng_    = noble_gas_fun(gases=['He'], E=self.E, T=self.T, Ae=self.Ae, F=self.F, P='lapse_rate')
        He4_eq  = ng_.equil_conc()['He'] # atmospheric equilibrium
        He4_ex  = ng_.ce_exc(add_eq_conc=False)['He'] # excess air component
        He4_atm = ng_.ce_exc(add_eq_conc=True)['He']  # total atmosheric origin -- equilibrium + excess air
        He4_ter = He4_obs - He4_atm # terrigenic 4He
        #He_del = 100*(He_ter/He_atm-1)
        #He4_del = 100*(He4_ter/He4_eq)
        He4_del = 100*(He4_ter/He4_atm)
        self.obs_dict_['He4_eq']  = He4_eq
        self.obs_dict_['He4_atm'] = He4_atm
        self.obs_dict_['He4_ter'] = He4_ter 
        self.obs_dict_['He4_del'] = He4_del
        #---
        # 3He mass balance -- Followin NG book pg.642
        Ratm = 1.384e-6  #3He/4He in atmosphere
        #Req  = 1.360e-6
        #Rterr = 2.e-8 # Cook Book
        
        S = 0.0 # Salinity
        CF = 4.021e14/(1-S)
        #He_trit  = He3_obs - Req*He_atm - Ratm*He_exc - Rterr*He_ter
        He3_trit  = He3_obs - (He4_obs-He4_ter)*Ratm + He4_eq*Ratm*(1-0.983) - He4_ter*Rterr
        TU = He3_trit * CF
        self.obs_dict_['He3_tu'] = TU
        
               
    def Ne_del(self):
        # Delta Ne in percent
        Ne_obs = self.obs_dict_['Ne']
        ng_ = ng.noble_gas_fun(gases=['Ne'], E=self.E, T=self.T, Ae=self.Ae, F=self.F, P='lapse_rate')
        Ne_atm = ng_.equil_conc()['Ne']
        del_Ne = (Ne_obs/Ne_atm - 1)*100
        self.obs_dict_['Ne_del'] = del_Ne
        #print (del_Ne)













"""
# moved to own script
#------
# Model Inversions



#--------------------------------
# Read in the Observation Data
#--------------------------------
dd_ = pd.read_excel('..//NobleGas_compile.xlsx', skiprows=1, index_col='SiteID', nrows=9)
dd  = dd_.copy()[['4He','Ne','Ar','Kr','Xe']]
dd.rename(columns={'4He':'He'}, inplace=True)
dd.dropna(inplace=True)
dd.index.name = ''

obs_dict_all = dd.T.to_dict()

well_elev = {'PLM1':2786.889893,
             'PLM6_2017':2759.569824,
             'PLM6_2021':2759.569824,
             'PLM7':2782.550049,
             'PLM8':2760.909912,
             'PLM9':2760.939941,
             'Shumway':2880.36,
             'RMBL':2879.0} # this one just google earth, not confident



# Parameter notes
# PLM6
#err_dict={'He':2.0, 'Ne':2.0, 'Ar':2.0, 'Kr':3.0, 'Xe':12.0}

# PLM8 and Shumway
#err_dict={'He':2.0, 'Ne':2.0, 'Ar':2.0, 'Kr':3.0, 'Xe':12.0}
#gases = ['Ne', 'Ar', 'Kr', 'Xe']
#lapse_slope     = -108.72 # m/C for model
#err_lapse_slope =  abs(lapse_slope*0.1)
#lapse_b     = 3263.7
#err_lapse_b = 217.0 / 1.96 
#par_nam = ['Ae', 'F', 'E', 'T']
#Amin = 1.0e-4
#Amax = 2.5e-1
#Fmin = 1.0e-3
#Fmax = 50.0e+0




#-----------------------
# MCMC input args
#-----------------------

well = 'PLM1'

bckend    = str(well)
Run_Calib = True 

obs_dict = obs_dict_all[well]

err_dict={'He':2.0, 'Ne':2.0, 'Ar':2.0, 'Kr':5.0, 'Xe':8.0}

gases = ['Ne', 'Ar', 'Kr', 'Xe']

lapse_slope     = -108.72 # m/C for model
err_lapse_slope =  abs(lapse_slope*0.1)
print ('lapse rate slope = {:.3f} (C/100m)'.format(1/lapse_slope * 100))
print ('intercept Delta for 2C change = {:.3f} (meters)'.format(lapse_slope*2))

lapse_b     = 3263.7
err_lapse_b = 217.0 / 1.96 #lapse_b*0.01

par_nam = ['Ae', 'F', 'E', 'T']

Amin = 1.0e-4
Amax = 2.5e-1

Fmin = 1.0e-3
Fmax = 3.0e+0

Emin = well_elev[well]-5.0 # Elevation of the well
#Emax = 2950.0 # Elevation of top of hillslope
Emax = 3400.0 # Elevation of top of Snodgrass

Tmin = find_T_lapse(Emax, lapse_b, lapse_slope)
Tmax = find_T_lapse(Emin, lapse_b, lapse_slope)




#---
# set up figure dirs
# Create a new directory for figures
fdir = '{}_figs'.format(well)
if os.path.exists(fdir) and os.path.isdir(fdir):
    pass
else:
    os.makedirs(fdir)




#---------------------- 
# MCMC Beta Priors
#----------------------
par_min = np.array([Amin, Fmin, Emin, Tmin]) #Ae, F, E, T
par_max = np.array([Amax, Fmax, Emax, Tmax]) #Ae, F, E, T
par_bnd = pd.DataFrame(data=np.column_stack((par_min,par_max)), index=par_nam, columns=['min','max'])
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

    T = theta[-1]
    Ae, F, E = frombeta(['Ae','F','E'], theta[:-1]) # convert beta priors to normal space

    ce_mod = noble_gas_fun(gases=gases, E=E, T=T, Ae=Ae, F=F, P='lapse_rate').ce_exc(add_eq_conc=True)
    ce_mod_ = np.array([ce_mod[i] for i in gases]) 
    return ce_mod_
    

def mcmc_model():
    model = mc.Model()
    with model:
        #pdb.set_trace()
        #--- 
        # Priors
        # Normal
        #Ae = mc.Normal('Ae', mu=0.001,  sigma=0.01)
        #F  = mc.Normal('F',  mu=2.0,    sigma=0.50)
        #E  = mc.Normal('E',  mu=2800.0, sigma=50.0)
        
        # Beta
        Ae = mc.Beta('Ae', alpha=2, beta=4)
        F  = mc.Beta('F',  alpha=2, beta=4)
        E  = mc.Beta('E',  alpha=2, beta=2)
        
        #m = lapse_slope
        #b = lapse_b
        m = mc.Normal('m', mu=lapse_slope, sigma=err_lapse_slope)
        b = mc.Normal('b', mu=lapse_b, sigma=err_lapse_b)
                
        # Find T using lapse rate and Elevation RV
        E_ = frombeta(['E'], [E])[0] # E needs to normal space, not beta
        T  = mc.Deterministic('T', find_T_lapse(E_, b, m))
                
        theta = tt.as_tensor_variable([Ae, F, E, T])
        #theta = tt.as_tensor_variable([Ae, F, E, T, m, b])
            
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
        like = mc.Normal('like', mu=mod_out, sd=obs_sd, observed=obs_mu)
    return model



#------------------
# Prior Predictive
#------------------
with mcmc_model():
    # This does not include Helium
    prior_checks = mc.sample_prior_predictive(samples=50000, random_seed=10)
    
    # Actual Parameters
    vrs = ['Ae','F','E'] 
    for p in vrs:
        prior_checks[p] = prior_checks[p] * (par_bnd_[p]['max']-par_bnd_[p]['min'])+par_bnd_[p]['min']
    vrs = ['Ae','F','E','T','m','b']
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8,6))
    for i in range(len(vrs)):
        r,c = i//3, i%3
        ax[r,c].hist(prior_checks[vrs[i]], bins=20)
        ax[r,c].axvline(prior_checks[vrs[i]].mean(), color='black', linestyle='--', zorder=10)
        ax[r,c].set_xlabel(vrs[i])
    fig.tight_layout()
    plt.show()
    
    
    # Concentrations
    pr = prior_checks['like']
    # plot to check
    fig, ax = plt.subplots(ncols=4, figsize=(8,6))
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
    ax[0].set_ylabel(r'Concentration (cm$^{3}$STP/g)')
    fig.suptitle('Prior Predictive')
    fig.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), fdir, 'prior_predictive.jpg'), dpi=320)
    plt.show()


    



#------------------------------------------------
# Sample the posterior
#------------------------------------------------
if Run_Calib:
    from shutil import rmtree
    if os.path.exists(bckend+'.netcdf'):
        #rmtree(bckend+'.netcdf')
        os.remove(bckend+'.netcdf')
        
    with mcmc_model():
        #prop_var = np.eye(par.shape[1]) * np.ones(par.shape[1])*0.1
        #prop_var = np.ones(par.shape[1])*0.25
        sampler = mc.DEMetropolisZ(tune_interval=10000)#, S=prop_var) #10000 works
        #sampler = mc.Metropolis()
        #sampler = mc.Slice()
        
        trace = mc.sample(tune=10000, draws=100000, step=sampler, chains=6, cores=1, random_seed=123423, 
                          idata_kwargs={'log_likelihood':False}, discard_tuned_samples=True)
        
        prior = mc.sample_prior_predictive(25000)
        post_pred = mc.sample_posterior_predictive(trace, samples=30000, random_seed=123423)
        idata = az.from_pymc3(trace, prior=prior, posterior_predictive=post_pred, log_likelihood=False)
    
        # Save 
        az.to_netcdf(idata, bckend+'.netcdf')
else:
    pass



#------------------------------------------------
# Plotting MCMC Results
#------------------------------------------------
with mcmc_model():
    # Read in data
    #if Run_Calib:
    #    pass
    #else:
    idata = az.from_netcdf(bckend+'.netcdf') 
 
    # Convert from beta parameters to normal
    #vrs = list(idata.posterior.keys())
    vrs = ['Ae','F','E'] 
    for p in vrs:
        idata.posterior[p] = idata.posterior[p] * (par_bnd_[p]['max']-par_bnd_[p]['min'])+par_bnd_[p]['min']
        idata.prior[p] = idata.prior[p] * (par_bnd_[p]['max']-par_bnd_[p]['min'])+par_bnd_[p]['min']
 
    
    trace_summary = az.summary(idata, round_to=6)
    print (trace_summary)
    
    # Traceplot
    az.plot_trace(idata)
    plt.savefig(os.path.join(os.getcwd(), fdir, 'trace_plot.jpg'), dpi=320)
    plt.show()
    
    # Posterior plot
    az.plot_posterior(idata, round_to=3)
    #plt.savefig('mcmc_posterior.png', dpi=300)
    plt.savefig(os.path.join(os.getcwd(), fdir, 'post.jpg'), dpi=320)
    plt.show()

    # Posterior with prior
    ax = az.plot_density([idata.posterior, idata.prior],
                    var_names=['Ae','F','E','T','m','b'],
                    data_labels=["Posterior", "Prior"],
                    shade=0.1,
                    hdi_prob=1.0,
                    textsize=14)
    plt.savefig(os.path.join(os.getcwd(), fdir, 'post_prior.jpg'), dpi=320)
    plt.show()

    
    # Joint posterior plots
    ax = az.plot_pair(idata, kind='hexbin', marginals=True, textsize=14)
    plt.savefig(os.path.join(os.getcwd(), fdir, 'post_joint.jpg'), dpi=320)
    plt.show()
    
    



#-------------------------
# Posterior predictive 
#-------------------------

gases_ = ['He', 'Ne', 'Ar', 'Kr', 'Xe']

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
C_max_ = noble_gas_fun(gases=gases_, E=E, T=T, Ae=Ae, F=F, P='lapse_rate').ce_exc(add_eq_conc=True)
C_max = np.array([C_max_[i] for i in gases_])
# Ensemble of predictions for uncertainty
C_unc = []
for i in range(75000):
    c = noble_gas_fun(gases=gases_, E=EE[i], T=TT[i], Ae=AA[i], F=FF[i], P='lapse_rate').ce_exc(add_eq_conc=True)
    cc = np.array([c[i] for i in gases_])
    C_unc.append(cc)
C_unc = np.array(C_unc)


#-----------
# Plot 2
# Post-predictive concentration box-plots
fig, ax = plt.subplots(ncols=5, figsize=(8,6))
for i in range(len(gases_)):
    # Observations
    ax[i].scatter(0.25, obs_arr[i], color='red', zorder=10)
    ax[i].errorbar(0.25, obs_arr[i], yerr=obs_err[i], color='red', zorder=10)
    # Maximum a posteriori
    # Maximum a posteriori 
    ax[i].scatter(-0.2, C_max[i], color='blue', marker='_', label=r'$\hat{C}_{mod}$')
    # Predictions using boxplots
    ax[i].boxplot(C_unc[:,i], positions=[0], showmeans=False, medianprops={'color':'black','linewidth':2})
    # Clean up
    ax[i].set_xticks([])
    ax[i].set_xticklabels([])
    ax[i].set_xlabel(gases_[i])
    ax[i].set_yscale('log')
    #ymin = np.floor(np.log10(ax[i].get_ylim()[0]))
    #ymax = np.ceil(np.log10(ax[i].get_ylim()[1]))
    #ax[i].set_ylim(10**ymin, 10**ymax)
ax[0].set_ylabel(r'Concentration (cm$^{3}$STP/g)')
fig.suptitle('{} Posterior Predictive'.format(well))
fig.tight_layout()
plt.savefig(os.path.join(os.getcwd(), fdir, 'post_conc_pred.jpg'), dpi=320)
plt.show()



#-----------
# Plot 3
# Posterior Predictive lapse rate
fig, ax = plt.subplots()
xx = np.linspace(-1, 8, 500)
# prior model
ax.plot(xx, lapse_slope*xx + lapse_b, c='C2', linewidth=3, zorder=5, label='Prior Model')
# maximum a posteriori
ax.plot(xx, trace_summary.loc['m','mean']*xx + trace_summary.loc['b','mean'], 
        linewidth=3, color='black', zorder=6, label='Max a Posteriori')
# error ensembles
for i in range(500):
    ax.plot(xx, MM[i]*xx + BB[i], c='grey', alpha=0.5, label='Posterior Uncertainty' if i==0 else '')
# Add inferred recharge zone
ax.scatter(trace_summary.loc['T','mean'], trace_summary.loc['E','mean'], zorder=8)
xmin = trace_summary.loc['T','mean']-trace_summary.loc['T','sd']
ymin = trace_summary.loc['E','mean']-trace_summary.loc['E','sd']
xsd  = trace_summary.loc['T','sd']*2
ysd  = trace_summary.loc['E','sd']*2
rect = patches.Rectangle((xmin, ymin), xsd, ysd, linewidth=1, edgecolor='C0', facecolor='none', zorder=10, label=r'Recharge Zone (1$\sigma$)')
ax.add_patch(rect)
# Add in well head loc?
ax.axhline(well_elev[well], color='black', linestyle='--', linewidth=1.0, label='Well Elevation')
ax.axhline(2950.0, color='black', linestyle=':', linewidth=1.0, label='Top Hillslope Elevation')
# Clean up
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Elevation (m)')
ax.yaxis.set_major_locator(MultipleLocator(250))
ax.yaxis.set_minor_locator(MultipleLocator(50))
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.grid(alpha=0.3)
ax.legend(loc='upper right')
fig.suptitle(well)
fig.tight_layout()
plt.savefig(os.path.join(os.getcwd(), fdir, 'post_laspe_pred.jpg'), dpi=320)
plt.show()
"""





