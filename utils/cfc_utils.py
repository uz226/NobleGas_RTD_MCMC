import numpy as np
import pandas as pd
import pickle
#import lmfit as lm
import scipy
import datetime
import pdb

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.ticker as ticker
plt.rcParams['font.size']=14
#plt.rcParams["text.usetex"] = False

#import convolution_integral_nt as conv





#-----------------------------------------------
# Correct CFC field observations for:
# excess air, temperature, elevation
#-----------------------------------------------
class cfc_ce_corr():
    def __init__(self, cfc_num, E, T, Ae, F, S=0.0):
        self.cfc = cfc_num      # list of cfc -- e.g [11,12,113]
        self.E   = E            # elevation in meters
        self.T   = T            # temperature in C
        self.Ae  = Ae*1000.     # initial volume of entrapped air in ccSTP/g
        self.F   = F            # fractionation factor for CE model
        self.S   = S            # salinity
        self.P   = None         # pressure in atm -- calculated from elevation using lapse_rate_atm()
       
    def vapor_pressure_atm(self):
        '''Returns:
                -- P_vapor, the vapor pressure in atm using the Antione equation.
           kwargs:
                -- Temperature in Celcius'''     
        if  self.T<=99.0:
            A=8.07131;
            B=1730.63;
            C=233.426;
        else:
            A=8.14019;
            B=1810.94;
            C=244.485;
        P = 10**(A-(B/(C+self.T))) #mmHG
        P = P/760.*101325          #Pa
        P_vapor=P/1.0e9            #GPa
        P_vapor = P_vapor/0.000101325 #atm
        return P_vapor
    
    def lapse_rate_atm(self):
        '''P = lapse_rate(Elev).  
        Returns: 
            -- Atmospheric pressure in atm
        kwargs:
            -- Elevation in meters
        Assumes a pressure of 1 atm at sea level.'''
        P = ((1-.0065*self.E/288.15)**5.2561) #atm  
        self.P = P
        return P
    
    def solubility_cfc(self):
        '''K_cfc = solubility_cfc11(T,S).
        Returns: 
            -- Solubility coefficient K in [mol atm^-1 kg^-1] 
            -- assumes C(mol/kg) = K*z_i*(P-Pw) (Bullister 2002)
        kwrags:
            -- Temperature in Celcius'''
        T_k = self.T+273.15
        K_ = []
        for cfc in self.cfc:
            cfc_dict = {11:[-136.2685,206.1150,57.2805,-.148598,0.095114,-0.0163396],12:[-124.4395,185.4299,51.6383,-0.149779,0.094668,-0.0160043],113:[-136.129,206.475,55.8957,-0.02754,0.006033,0,]}
            a_1 = cfc_dict[cfc][0]
            a_2 = cfc_dict[cfc][1]
            a_3 = cfc_dict[cfc][2]
            b_1 = cfc_dict[cfc][3]
            b_2 = cfc_dict[cfc][4]
            b_3 = cfc_dict[cfc][5]
            K = np.exp(a_1+a_2*(100/T_k)+a_3*np.log(T_k/100)+self.S*(b_1+b_2*(T_k/100)+b_3*(T_k/100)**2))
            K_.append(K)
        return np.array(K_)
    
    def equil_air_conc_cfc(self, C_meas):
        '''
        Convert measured aqeuous CFC into atmospheric mixing ratio, with effect of excess air, temperature, and pressure.
        If excess air componenets go to zero, this returns the equilibrium component corrected for temperature and pressure
        
        Returns: 
            -- z_i, the equilibrium air concentration (mol/mol or pptv)
        Inputs:
            -- C_meas: the measured aq. concentration in pmol/kg
        kwargs:
            -- T_rech: recharge temperature in celcius
            -- E_rech: recharge elevation in meters
            -- Ae:     excess air in ccSTP/kg
            -- F:      fraction factor (see Aesbhac-Hertig 2002)'''
        Ki = self.solubility_cfc()  #mol atm^-1 l^-1
            
        P_da = self.lapse_rate_atm()-self.vapor_pressure_atm()
        molar_volume = 22414.1
        
        z_i = (C_meas + ((C_meas*self.F*(self.Ae/molar_volume))/(Ki*P_da))) / (Ki*P_da+(self.Ae/molar_volume))
        return z_i
    
    def equil_aq_conc_cfc(self, z_i):
        '''
        Convert atmospheric mixing ratio to aqueous concentration; C = K*z_i*(P-Pw) 
        Returns:
            -- Ci, the aqueous equilibrium concentration of CFC in pmol/kg 
        Inputs:
            -- z_i: atmospheric mixing ratio in pptv
        kwargs:
            -- T_rech: recharge temperature in celcius
            -- E_rech: recharge elevation in meters
        Note: could use this function to generate site-specific input function'''
        #pdb.set_trace()
        Ki = self.solubility_cfc()  #mol atm^-1 l^-1        
        P_da = self.lapse_rate_atm()-self.vapor_pressure_atm()
        
        C = Ki*z_i*P_da #pmol/kg (because z_i in pptv)
        return C
    
    def ce_exc_conc_cfc(self, z_i):
        '''
        Returns:
            -- Concentration of cfc in [pmol/kg] due to excess air
        Inputs:
            -- z_i: air mixing ratios in pptv
        kwargs:
            -- T:   temperature in celcius, 
            -- S:   salinity in parts per thousand 
            -- P:   atmospheric pressure in atm
            -- Ae:  excess air in ccSTP/kg
            -- F:   Fraction factor (see Aesbhac-Hertig 2002)'''
        P_vapor = self.vapor_pressure_atm() #atm
        Kcfc = self.solubility_cfc()  #mol atm^-1 l^-1
        C_eq = Kcfc*z_i*(self.P-P_vapor) #[mol/kg] (because z_i in pptv)
        Ae = self.Ae/22414/1e-12 #Ae in mol/kg the -- 1e12 mixing ratio in pptv
        C_exc = ((1-self.F)*Ae*(z_i*1e-12))/(1+self.F*Ae*((z_i*1e-12/C_eq))) # in mol/kg
        return C_exc
    
    def update_pars(self, T, E, Ae, F):
        self.T  = T
        self.E  = E
        self.Ae = Ae*1000
        self.F  = F
        self.vapor_pressure_atm()
        self.lapse_rate_atm()
    
 
    
 
    
#-----------------------------------------------
# Correct SF6 field Excess air and Temperature
#-----------------------------------------------
class sf6_ce_corr():
    def __init__(self, E, T, Ae, F, S=0.0):
        self.E   = E            # elevation in meters
        self.T   = T            # temperature in C
        self.Ae  = Ae*1000.     # initial volume of entrapped air, ccSTP/g
                                # convert from ccSTP/g to ccSTP/kg for below calculations
        self.F   = F            # fractionation factor for CE model
        self.S   = S            # salinity
        self.P   = None         # Pressure in atm
       
    def vapor_pressure_atm(self):
        '''Returns:
                -- P_vapor, the vapor pressure in atm using the Antione equation.'''     
        if  self.T<=99.0:
            A=8.07131;
            B=1730.63;
            C=233.426;
        else:
            A=8.14019;
            B=1810.94;
            C=244.485;
                
        P = 10**(A-(B/(C+self.T))) #mmHG
        P = P/760.*101325     #Pa
        P_vapor=P/1.0e9       #GPa
        P_vapor = P_vapor/0.000101325 #atm
        return P_vapor
    
    def lapse_rate_atm(self):
        '''P = lapse_rate(Elev).  
        Returns: 
            -- Atmospheric pressure in atm
        Assumes a pressure of 1 atm at sea level.'''
        P = ((1-.0065*self.E/288.15)**5.2561) #atm  
        self.P = P
        return P
    
    def solubility_sf6(self):
        '''K_sf6 = solubility_sf6(T,S). 
        Returns: 
            -- Solubility coefficient in mol atm^-1 kg^-1 
            -- assumes C(mol/kg) = K*z_i*(P-Pw) (Bullister 2002)'''
        T_k = self.T+273.15;
        a_1 = -98.7264000;
        a_2 = 142.803;
        a_3 = 38.8746;
        b_1 = 0.0268696;
        b_2 = -0.0334407;
        b_3 = 0.0070843;
        K = np.exp(a_1+a_2*(100/T_k)+a_3*np.log(T_k/100)+self.S*(b_1+b_2*(T_k/100)+b_3*(T_k/100)**2));
        return K
    
    #def equil_air_conc_sf6(self, C_meas):
    #    '''
    #    Returns: 
    #        -- z_i, the equilibrium air concentration (unitless or pptv)
    #    Inputs:
    #        -- C_meas: the measured concentration in pmol/kg
    #        -- T_rech: recharge temperature in celcius
    #        -- E_rech: recharge elevation in meters
    #        -- Ae:     excess air in ccSTP/kg
    #        -- F:      Fraction factor (see Aesbhac-Hertig 2002)'''
    #    Ki = self.solubility_sf6()
    #    P_da = self.lapse_rate_atm()-self.vapor_pressure_atm();
    #    molar_volume = 22414.1
    #    z_i = (C_meas + ((C_meas*F*(self.Ae/molar_volume))/(Ki*P_da)))/(Ki*P_da+(self.Ae/molar_volume))*.001
    #    return z_i


    def equil_air_conc_sf6(self, C_meas):
        '''
        Convert measured aqeuous CFC into atmospheric mixing ratio, with effect of excess air, temperature, and pressure.
        If excess air componenets go to zero, this returns the equilibrium component corrected for temperature and pressure
        
        Returns: 
            -- z_i, the equilibrium air concentration (mol/mol or pptv)
        Inputs:
            -- C_meas: the measured aq. concentration in pmol/kg
        kwargs:
            -- T_rech: recharge temperature in celcius
            -- E_rech: recharge elevation in meters
            -- Ae:     excess air in ccSTP/kg
            -- F:      fraction factor (see Aesbhac-Hertig 2002)'''
        Ki = self.solubility_sf6()  #mol atm^-1 l^-1
            
        P_da = self.lapse_rate_atm()-self.vapor_pressure_atm()
        molar_volume = 22414.1
        
        z_i = (C_meas + ((C_meas*self.F*(self.Ae/molar_volume))/(Ki*P_da))) / (Ki*P_da+(self.Ae/molar_volume))*0.001
        return z_i


    #def equil_aq_conc_sf6(self, z_i):
    #    '''C = equil_conc_sf6(T,z,S=0,P=1.) 
    #    Returns:
    #        -- Equilibrium concentration of SF6 in fmol/kg
    #    Inputs:
    #        -- z_i: atmospheric mixing ratio in pptv
    #    C = K*z_i*(P-Pw)'''
    #    P_vapor = self.vapor_pressure_atm();
    #    Ksf6 = self.solubility_sf6()  #mol atm^-1 l^-1
    #    C = Ksf6*z_i*(self.P-P_vapor)/.001 #fmol/kg (because z_i in pptv)
    #    return C
      
    def equil_aq_conc_sf6(self, z_i):
        '''
        Convert atmospheric mixing ratio to aqueous concentration; C = K*z_i*(P-Pw) 
        Returns:
            -- Ci, the aqueous equilibrium concentration of CFC in pmol/kg 
        Inputs:
            -- z_i: atmospheric mixing ratio in pptv
        kwargs:
            -- T_rech: recharge temperature in celcius
            -- E_rech: recharge elevation in meters
        Note: could use this function to generate site-specific input function'''
        #pdb.set_trace()
        Ki = self.solubility_sf6()  #mol atm^-1 l^-1        
        P_da = self.lapse_rate_atm()-self.vapor_pressure_atm()
        
        C = Ki*z_i*P_da/0.001 #fmol/kg (because z_i in pptv)
        return C  
      
    def ce_exc_conc_sf6(self, z_i):
        '''
        Returns:
            -- Concentration of sf6 in [fmol/kg] due to excess air
        Inputs:
            -- T:   temperature in celcius, 
            -- S:   salinity in parts per thousand 
            -- P:   atmospheric pressure in atm
            -- z_i: air mixing ratios in pptv
            -- Ae:  excess air in ccSTP/kg
            -- F:   Fraction factor (see Aesbhac-Hertig 2002)'''
        P_vapor = self.vapor_pressure_atm() #atm
        Ksf6 = self.solubility_sf6()  #mol atm^-1 l^-1
        C_eq = Ksf6*z_i*(self.P-P_vapor) #[mol/kg] (because z_i in pptv)
        Ae = self.Ae/22414/1e-12   #Ae in mol/kg -- 1e-12 because mixing ration in pptv
        C_exc = 1000 * ((1-self.F)*Ae*(z_i*1e-12))/(1+self.F*Ae*((z_i*1e-12/C_eq))) # in fmol/kg
        return C_exc  
    
    def update_pars(self, T, E, Ae, F):
        self.T  = T
        self.E  = E
        self.Ae = Ae*1000
        self.F  = F
        self.vapor_pressure_atm()
        self.lapse_rate_atm()
    
    

    
    

